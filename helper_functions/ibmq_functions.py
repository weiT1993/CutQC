from qiskit import IBMQ
from qiskit.providers.jobstatus import JobStatus
from qiskit.providers.aer.noise import NoiseModel
from qiskit.transpiler import CouplingMap
from datetime import timedelta, datetime
from pytz import timezone
import time
import subprocess
import os
import pickle

from helper_functions.non_ibmq_functions import read_dict


def load_IBMQ(token, hub, group, project):
    IBMQ.save_account(token, overwrite=True)
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub=hub, group=group, project=project)
    return provider


def get_device_info(token, hub, group, project, device_name, fields, datetime):
    dirname = "./devices/%s" % datetime.date()
    filename = "%s/%s.pckl" % (dirname, device_name)
    _device_info = read_dict(filename=filename)
    if len(_device_info) == 0:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        else:
            subprocess.run(["rm", "-r", dirname])
            os.makedirs(dirname)
        provider = load_IBMQ(token=token, hub=hub, group=group, project=project)
        for x in provider.backends():
            if "simulator" not in str(x):
                device = provider.get_backend(str(x))
                properties = device.properties(datetime=datetime)
                num_qubits = device.configuration().n_qubits
                print("Download device_info for %d-qubit %s" % (num_qubits, x))
                coupling_map = CouplingMap(device.configuration().coupling_map)
                noise_model = NoiseModel.from_backend(device)
                basis_gates = device.configuration().basis_gates
                _device_info = {
                    "properties": properties,
                    "coupling_map": coupling_map,
                    "noise_model": noise_model,
                    "basis_gates": basis_gates,
                }
                pickle.dump(_device_info, open("%s/%s.pckl" % (dirname, str(x)), "wb"))
            print("-" * 50)
        _device_info = read_dict(filename=filename)
    device_info = {}
    for field in fields:
        if field == "device":
            provider = load_IBMQ(token=token, hub=hub, group=group, project=project)
            device = provider.get_backend(device_name)
            device_info[field] = device
        else:
            device_info[field] = _device_info[field]
    return device_info, filename


def check_jobs(token, hub, group, project, cancel_jobs):
    provider = load_IBMQ(token=token, hub=hub, group=group, project=project)

    time_now = datetime.now(timezone("EST"))
    delta = timedelta(
        days=1, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0
    )
    time_delta = time_now - delta

    for x in provider.backends():
        if "qasm" not in str(x):
            device = provider.get_backend(str(x))
            properties = device.properties()
            num_qubits = len(properties.qubits)
            print(
                "%s: %d-qubit, max %d jobs * %d shots"
                % (
                    x,
                    num_qubits,
                    x.configuration().max_experiments,
                    x.configuration().max_shots,
                )
            )
            jobs_to_cancel = []
            print("QUEUED:")
            print_ctr = 0
            for job in x.jobs(limit=50, status=JobStatus["QUEUED"]):
                if print_ctr < 5:
                    print(
                        job.creation_date(),
                        job.status(),
                        job.queue_position(),
                        job.job_id(),
                        "ETA:",
                        job.queue_info().estimated_complete_time - time_now,
                    )
                jobs_to_cancel.append(job)
                print_ctr += 1
            print("RUNNING:")
            for job in x.jobs(limit=5, status=JobStatus["RUNNING"]):
                print(job.creation_date(), job.status(), job.queue_position())
                jobs_to_cancel.append(job)
            print("DONE:")
            for job in x.jobs(
                limit=5, status=JobStatus["DONE"], start_datetime=time_delta
            ):
                print(
                    job.creation_date(), job.status(), job.error_message(), job.job_id()
                )
            print("ERROR:")
            for job in x.jobs(
                limit=5, status=JobStatus["ERROR"], start_datetime=time_delta
            ):
                print(
                    job.creation_date(), job.status(), job.error_message(), job.job_id()
                )
            if cancel_jobs and len(jobs_to_cancel) > 0:
                for i in range(3):
                    print("Warning!!! Cancelling jobs! %d seconds count down" % (3 - i))
                    time.sleep(1)
                for job in jobs_to_cancel:
                    print(
                        job.creation_date(),
                        job.status(),
                        job.queue_position(),
                        job.job_id(),
                    )
                    job.cancel()
                    print("cancelled")
            print("-" * 100)
