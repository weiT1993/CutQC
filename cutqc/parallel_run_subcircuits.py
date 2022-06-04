import argparse, pickle, subprocess

from helper_functions.non_ibmq_functions import evaluate_circ

from cutqc.evaluator import modify_subcircuit_instance, mutate_measurement_basis, measure_prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_folder', metavar='S', type=str)
    parser.add_argument('--subcircuit_idx', metavar='N', type=int)
    parser.add_argument('--rank', metavar='N', type=int)
    args = parser.parse_args()

    subcircuit_idx = args.subcircuit_idx
    meta_info = pickle.load(open('%s/meta_info.pckl'%(args.data_folder),'rb'))
    subcircuit = meta_info['subcircuits'][subcircuit_idx]
    eval_mode = meta_info['eval_mode']
    num_shots_fn = meta_info['num_shots_fn']
    instance_init_meas_ids = meta_info['instance_init_meas_ids'][subcircuit_idx]
    rank_jobs = pickle.load(open('%s/rank_%d.pckl'%(args.data_folder,args.rank),'rb'))

    uniform_p = 1/2**subcircuit.num_qubits
    num_shots = num_shots_fn(subcircuit) if num_shots_fn is not None else None

    for instance_init_meas in rank_jobs:
        if 'Z' in instance_init_meas[1]:
            continue
        subcircuit_instance = modify_subcircuit_instance(
            subcircuit=subcircuit,
            init=instance_init_meas[0],meas=instance_init_meas[1])
        if eval_mode=='runtime':
            subcircuit_inst_prob = uniform_p
        elif eval_mode=='sv':
            subcircuit_inst_prob = evaluate_circ(circuit=subcircuit_instance,backend='statevector_simulator')
        elif eval_mode=='qasm':
            subcircuit_inst_prob = evaluate_circ(circuit=subcircuit_instance,backend='noiseless_qasm_simulator',options={'num_shots':num_shots})
        else:
            raise NotImplementedError
        mutated_meas = mutate_measurement_basis(meas=instance_init_meas[1])
        for meas in mutated_meas:
            measured_prob = measure_prob(unmeasured_prob=subcircuit_inst_prob,meas=meas)
            instance_init_meas_id = instance_init_meas_ids[(instance_init_meas[0],meas)]
            # print('%s --> rank %d writing subcircuit_%d_instance_%d'%(args.data_folder,args.rank,subcircuit_idx,instance_init_meas_id))
            pickle.dump(measured_prob,open('%s/subcircuit_%d_instance_%d.pckl'%(args.data_folder,subcircuit_idx,instance_init_meas_id),'wb'))
    subprocess.run(['rm','%s/rank_%d.pckl'%(args.data_folder,args.rank)])