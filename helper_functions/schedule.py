"""
Job submission/simulating backend
Input:
circ_dict (dict): circ (not transpiled), shots, evaluator_info (optional)
"""

import math, copy, random, pickle
import numpy as np
from qiskit.compiler import transpile, assemble
from qiskit import Aer, execute
from qiskit.providers.aer import QasmSimulator
from time import time
from datetime import datetime

from helper_functions.non_ibmq_functions import apply_measurement
from helper_functions.ibmq_functions import get_device_info
from helper_functions.conversions import dict_to_array, memory_to_dict


class ScheduleItem:
    def __init__(self, max_experiments, max_shots):
        self.max_experiments = max_experiments
        self.max_shots = max_shots
        self.circ_list = []
        self.shots = 0
        self.total_circs = 0

    def update(self, key, circ, shots):
        reps_vacant = self.max_experiments - self.total_circs
        if reps_vacant > 0:
            circ_shots = max(shots, self.shots)
            circ_shots = min(circ_shots, self.max_shots)
            total_reps = math.ceil(shots / circ_shots)
            reps_to_add = min(total_reps, reps_vacant)
            circ_list_item = {"key": key, "circ": circ, "reps": reps_to_add}
            self.circ_list.append(circ_list_item)
            self.shots = circ_shots
            self.total_circs += reps_to_add
            shots_remaining = shots - reps_to_add * self.shots
        else:
            shots_remaining = shots
        return shots_remaining


class Scheduler:
    # TODO: rewrite the interface
    def __init__(self, circ_dict, verbose):
        self.verbose = verbose
        self.circ_dict = circ_dict
        self.jobs = {}
        self.ibmq_schedules = {}

    def add_ibmq(self, token, hub, group, project):
        """
        Have to run this function first before submitting jobs to IBMQ or using noisy simulations
        """
        self.token = token
        self.hub = hub
        self.group = group
        self.project = project

    def submit_ibmq_jobs(self, device_names, transpilation, real_device):
        self.device_names = device_names
        for device_name in device_names:
            if self.verbose:
                print(
                    "-->",
                    "IBMQ Scheduler : Submitting %s Jobs" % device_name,
                    "<--",
                    flush=True,
                )

            today = datetime.now()
            device_info = get_device_info(
                token=self.token,
                hub=self.hub,
                group=self.group,
                project=self.project,
                device_name=device_name,
                fields=[
                    "device",
                    "properties",
                    "basis_gates",
                    "coupling_map",
                    "noise_model",
                ],
                datetime=today,
            )
            device_size = len(device_info["properties"].qubits)
            self._check_input(device_size=device_size)

            self.ibmq_schedules[device_name] = self._get_ibmq_schedule(
                device_max_shots=device_info["device"].configuration().max_shots,
                device_max_experiments=device_info["device"]
                .configuration()
                .max_experiments,
            )

            jobs = []
            for idx, schedule_item in enumerate(self.ibmq_schedules[device_name]):
                job_circuits = []
                for element in schedule_item.circ_list:
                    key = element["key"]
                    circ = element["circ"]
                    reps = element["reps"]
                    # print('Key {}, {:d} qubit circuit * {:d} reps'.format(key,len(circ.qubits),reps))

                    if circ.num_clbits == 0:
                        qc = apply_measurement(circuit=circ, qubits=circ.qubits)
                    else:
                        qc = circ

                    if transpilation:
                        mapped_circuit = transpile(qc, backend=device_info["device"])
                    else:
                        mapped_circuit = qc

                    self.circ_dict[key]["%s_circuit" % device_name] = mapped_circuit

                    circs_to_add = [mapped_circuit] * reps
                    job_circuits += circs_to_add

                assert len(job_circuits) == schedule_item.total_circs

                qobj = assemble(
                    job_circuits,
                    backend=device_info["device"],
                    shots=schedule_item.shots,
                    memory=True,
                )
                if real_device:
                    ibmq_job = device_info["device"].run(qobj)
                else:
                    ibmq_job = Aer.get_backend("qasm_simulator").run(qobj)
                jobs.append(ibmq_job)
                if self.verbose:
                    print(
                        "Submitting job {:d}/{:d} {} --> {:d} distinct circuits, {:d} * {:d} shots".format(
                            idx + 1,
                            len(self.ibmq_schedules[device_name]),
                            ibmq_job.job_id(),
                            len(schedule_item.circ_list),
                            len(job_circuits),
                            schedule_item.shots,
                        ),
                        flush=True,
                    )
            self.jobs[device_name] = jobs

    def retrieve_jobs(self, force_prob, save_memory, save_directory):
        for device_name in self.device_names:
            if self.verbose:
                print("-->", "IBMQ Scheduler : Retrieving %s Jobs" % device_name, "<--")
            jobs = self.jobs[device_name]
            assert len(self.ibmq_schedules[device_name]) == len(jobs)
            memories = {}
            for job_idx in range(len(jobs)):
                schedule_item = self.ibmq_schedules[device_name][job_idx]
                hw_job = jobs[job_idx]
                if self.verbose:
                    print(
                        "Retrieving job {:d}/{:d} {} --> {:d} circuits, {:d} * {:d} shots".format(
                            job_idx + 1,
                            len(jobs),
                            hw_job.job_id(),
                            len(schedule_item.circ_list),
                            schedule_item.total_circs,
                            schedule_item.shots,
                        ),
                        flush=True,
                    )
                ibmq_result = hw_job.result()
                start_idx = 0
                for element_ctr, element in enumerate(schedule_item.circ_list):
                    key = element["key"]
                    circ = element["circ"]
                    reps = element["reps"]
                    end_idx = start_idx + reps
                    # print('{:d}: getting {:d}-{:d}/{:d} circuits, key {} : {:d} qubit'.format(element_ctr,start_idx,end_idx-1,schedule_item.total_circs-1,key,len(circ.qubits)),flush=True)
                    for result_idx in range(start_idx, end_idx):
                        ibmq_memory = ibmq_result.get_memory(result_idx)
                        if key in memories:
                            memories[key] += ibmq_memory
                        else:
                            memories[key] = ibmq_memory
                    start_idx = end_idx

            process_begin = time()
            counter = 0
            log_counter = 0
            for key in self.circ_dict:
                iteration_begin = time()
                full_circ = self.circ_dict[key]["circuit"]
                shots = self.circ_dict[key]["shots"]
                ibmq_memory = memories[key][:shots]
                mem_dict = memory_to_dict(memory=ibmq_memory)
                hw_prob = dict_to_array(
                    distribution_dict=mem_dict, force_prob=force_prob
                )
                self.circ_dict[key]["%s|hw" % device_name] = copy.deepcopy(hw_prob)
                if save_memory:
                    self.circ_dict[key]["%s_memory" % device_name] = copy.deepcopy(
                        ibmq_memory
                    )
                # print('Key {} has {:d} qubit circuit, hw has {:d}/{:d} shots'.format(key,len(full_circ.qubits),sum(hw.values()),shots))
                # print('Expecting {:d} shots, got {:d} shots'.format(shots,sum(mem_dict.values())),flush=True)
                if len(full_circ.clbits) > 0:
                    assert len(self.circ_dict[key]["%s|hw" % device_name]) == 2 ** len(
                        full_circ.clbits
                    )
                else:
                    assert len(self.circ_dict[key]["%s|hw" % device_name]) == 2 ** len(
                        full_circ.qubits
                    )
                if save_directory is not None:
                    pickle.dump(
                        self.circ_dict[key],
                        open("%s/%s.pckl" % (save_directory, key), "wb"),
                    )
                counter += 1
                log_counter += time() - iteration_begin
                if log_counter > 60 and self.verbose:
                    elapsed = time() - process_begin
                    eta = elapsed / counter * len(self.circ_dict) - elapsed
                    print(
                        "Processed %d/%d circuits, elapsed = %.3e, ETA = %.3e"
                        % (counter, len(self.circ_dict), elapsed, eta),
                        flush=True,
                    )
                    log_counter = 0

    def run_simulation_jobs(self, device_name):
        """
        device_name: 'noiseless' - noiseless simulation, 'IBMQ_XXX' - noisy simulation with IBMQ device noise model

        noiseless: run the circuits as is
        IBMQ_XXX: transpile
        """
        if self.verbose:
            print(
                "-->",
                "IBMQ Scheduler : Run %s Simulations" % device_name,
                "<--",
                flush=True,
            )
        self._check_input(device_size=None)

        if "ibmq" in device_name:
            today = datetime.now()
            device_info = get_device_info(
                token=self.token,
                hub=self.hub,
                group=self.group,
                project=self.project,
                device_name=device_name,
                fields=["device", "noise_model"],
                datetime=today,
            )
            noise_model = device_info["noise_model"]
        elif device_name == "noiseless":
            noise_model = None
        else:
            raise NotImplementedError

        simulation_begin = time()
        log_counter = 0
        counter = 0
        for key in self.circ_dict:
            iteration_begin = time()
            value = self.circ_dict[key]

            if value["circuit"].num_clbits == 0:
                qc = apply_measurement(
                    circuit=value["circuit"], qubits=value["circuit"].qubits
                )
            else:
                qc = value["circuit"]

            if "ibmq" in device_name:
                mapped_circuit = transpile(qc, backend=device_info["device"])
            elif device_name == "noiseless":
                mapped_circuit = qc
            self.circ_dict[key]["mapped_circuit"] = mapped_circuit

            simulation_result = execute(
                value["mapped_circuit"],
                Aer.get_backend("qasm_simulator"),
                noise_model=noise_model,
                shots=value["shots"],
            ).result()

            counts = simulation_result.get_counts(0)
            counts = dict_to_array(distribution_dict=counts, force_prob=True)
            self.circ_dict[key]["%s|sim" % device_name] = counts

            log_counter += time() - iteration_begin
            elapsed = time() - simulation_begin
            counter += 1
            if log_counter > 300 and self.verbose:
                eta = elapsed / counter * len(self.circ_dict) - elapsed
                print(
                    "Simulated %d/%d circuits, elapsed = %.3f, ETA = %.3f"
                    % (counter, len(self.circ_dict), elapsed, eta)
                )
                log_counter = 0

    def _check_input(self, device_size):
        assert isinstance(self.circ_dict, dict)
        for key in self.circ_dict:
            value = self.circ_dict[key]
            if "circuit" not in value or "shots" not in value:
                raise Exception(
                    "Input circ_dict should have `circuit`, `shots` for key {}".format(
                        key
                    )
                )
            elif device_size is not None and value["circuit"].num_qubits > device_size:
                raise Exception(
                    "Input circuit for key {} has {:d}-q ({:d}-q device)".format(
                        key, value["circuit"].num_qubits, device_size
                    )
                )

    def _get_ibmq_schedule(self, device_max_shots, device_max_experiments):
        circ_dict = copy.deepcopy(self.circ_dict)
        ibmq_schedule = []
        schedule_item = ScheduleItem(
            max_experiments=device_max_experiments, max_shots=device_max_shots
        )
        key_idx = 0
        while key_idx < len(circ_dict):
            key = list(circ_dict.keys())[key_idx]
            circ = circ_dict[key]["circuit"]
            shots = circ_dict[key]["shots"]
            # print('adding %d qubit circuit with %d shots to job'%(len(circ.qubits),shots))
            shots_remaining = schedule_item.update(key, circ, shots)
            if shots_remaining > 0:
                ibmq_schedule.append(schedule_item)
                schedule_item = ScheduleItem(
                    max_experiments=device_max_experiments, max_shots=device_max_shots
                )
                circ_dict[key]["shots"] = shots_remaining
            else:
                circ_dict[key]["shots"] = shots_remaining
                key_idx += 1
        if schedule_item.total_circs > 0:
            ibmq_schedule.append(schedule_item)
        return ibmq_schedule
