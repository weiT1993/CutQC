import pickle, glob
import matplotlib.pyplot as plt
import numpy as np

from example import make_QV

from qiskit_helper_functions.metrics import chi2_distance, MSE, MAPE, cross_entropy, HOP
from qiskit_helper_functions.non_ibmq_functions import evaluate_circ, read_dict

if __name__ == '__main__':
    circuit = make_QV()
    ground_truth = evaluate_circ(circuit=circuit,backend='statevector_simulator')
    no_cutting_metrics = {}
    for num_shots in [1024,8192,65536]:
        no_cutting_metrics[num_shots] = {
            'chi2':np.array([]),
            'Mean Squared Error':np.array([]),
            'Mean Absolute Percentage Error':np.array([]),
            'Cross Entropy':np.array([]),
            'HOP':np.array([])
        }
        for trial in range(10):
            qasm_sim = evaluate_circ(circuit=circuit,backend='noiseless_qasm_simulator',options={'num_shots':num_shots})
            no_cutting_metrics[num_shots]['chi2'] = np.append(no_cutting_metrics[num_shots]['chi2'],chi2_distance(target=ground_truth,obs=qasm_sim))
            no_cutting_metrics[num_shots]['Mean Squared Error'] = np.append(no_cutting_metrics[num_shots]['Mean Squared Error'],MSE(target=ground_truth,obs=qasm_sim))
            no_cutting_metrics[num_shots]['Mean Absolute Percentage Error'] = np.append(no_cutting_metrics[num_shots]['Mean Absolute Percentage Error'],MAPE(target=ground_truth,obs=qasm_sim))
            no_cutting_metrics[num_shots]['Cross Entropy'] = np.append(no_cutting_metrics[num_shots]['Cross Entropy'],cross_entropy(target=ground_truth,obs=qasm_sim))
            no_cutting_metrics[num_shots]['HOP'] = np.append(no_cutting_metrics[num_shots]['HOP'],HOP(target=ground_truth,obs=qasm_sim))

    circuit_folders = glob.glob('./cutqc_data/largest_QV_*_shots_trial_*')
    for metric_name in ['chi2','Mean Squared Error','Mean Absolute Percentage Error','Cross Entropy','HOP']:
        plt.figure()
        for quasi_conversion_mode in ['nearest','naive']:
            line_data = {}
            for folder in circuit_folders:
                num_shots = int(folder.split('./cutqc_data/largest_QV_')[1].split('_shots')[0])
                trial_metrics = read_dict(filename='%s/qc_8/qasm_24_1/metrics.pckl'%folder)
                trial_metric_result = trial_metrics[quasi_conversion_mode][metric_name]
                if num_shots not in line_data:
                    line_data[num_shots] = np.array([])
                line_data[num_shots] = np.append(line_data[num_shots],trial_metric_result)
            x_vals = []
            y_vals = []
            y_errs = []
            for num_shots in line_data:
                x_vals.append(num_shots)
                y_vals.append(np.mean(line_data[num_shots]))
                y_errs.append(np.std(line_data[num_shots]))
            x_vals, y_vals, y_errs = zip(*sorted(zip(x_vals, y_vals, y_errs)))
            plt.errorbar(x_vals,y_vals,yerr=y_errs,label=quasi_conversion_mode)
        x_vals = list(sorted(no_cutting_metrics.keys()))
        y_vals = [np.mean(no_cutting_metrics[x][metric_name]) for x in x_vals]
        y_errs = [np.std(no_cutting_metrics[x][metric_name]) for x in x_vals]
        plt.errorbar(x_vals,y_vals,yerr=y_errs,label='no cutting')
        plt.legend()
        plt.title(metric_name)
        plt.xlabel('shots')
        plt.ylabel(metric_name)
        plt.savefig('%s.pdf'%metric_name,dpi=400)
        plt.close()