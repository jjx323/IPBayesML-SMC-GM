### calculate samples mean, variance, trace plot, autocorrection
import numpy as np
import matplotlib.pyplot as plt
import scipy
from pathlib import Path


class SamplesAnalysis:
    def __init__(self, comm, path, idxs=[0], single_file=False, start_idx=0, end_idx=None, split_sign="_", format="npy"):
        self.path = Path(path)
        self.format = format
        self.split_sign = split_sign
        self.single_file = single_file
        self.comm = comm
        self.idxs = idxs
        if self.single_file is False:
            self.acquire_file_names(format=format)
            self.start_idx = start_idx
            if end_idx is None: end_idx = self.num_files
            num_total_idx = end_idx - start_idx
            assert self.num_files >= comm.size
            self.params_each = []
            if num_total_idx >= comm.size:
                num_ = num_total_idx//comm.size
            num_res_ = int(num_total_idx) % int(comm.size)
            assert num_res_ == 0, "num_res = " + str(num_res_) + "; num_files = " + str(num_total_idx) +  "; comm.size = " + str(self.comm.size)
            start_num, end_num = self.comm.rank * num_, (self.comm.rank + 1) * num_ - 1
            self.file_names = self.file_names[start_num:end_num+1]
            print('file name is',self.file_names)
        else:
            self.num_files = 1
            self.file_names = path

    def _sort(self, file_name):
        num = file_name.split(".")[0].split(self.split_sign)[-1]
        return np.int64(num)

    def acquire_file_names(self, format="npy"):
        self.data_files = self.path.glob("*." + format)
        self.file_names = sorted([file.name for file in self.data_files], key=self._sort)
        self.num_files = len(self.file_names)

    def eval_mean(self):
        num_data_local = 0
        num_files = len(self.file_names)
        for ii, file_name in enumerate(self.file_names):
            data = np.load(self.path / file_name)
            num_data_local += data.shape[0]
            if ii == 0:
                tmp = np.empty((num_files, data.shape[1]))
                fun_num_dof = data.shape[1]
            assert data.shape[1] == fun_num_dof
            tmp[ii, :] = np.sum(data, axis=0)
            print(file_name)
        average_vec_local = np.sum(tmp, axis=0) / num_data_local

        average_vec = self.comm.gather(average_vec_local, root=0)
        num_data = self.comm.gather(num_data_local, root=0, )
        if self.comm.rank == 0:
            mean_vec = np.average(np.array(average_vec), axis=0)
            num_data = np.sum(num_data)
            return mean_vec, num_data
        else:
            return None, None

    def eval_trace(self):
        for ii, file_name in enumerate(self.file_names):
            data = np.load(self.path / file_name)
            num_data_local = data.shape[0]
            if ii == 0:
                trace = np.empty((len(self.idxs), len(self.file_names) * num_data_local))
                fun_num_dof = data.shape[1]
            assert data.shape[1] == fun_num_dof
            trace[:, ii * num_data_local:(ii + 1) * num_data_local] = data[:, self.idxs].T
            print(file_name)

        trace_all = self.comm.gather(trace, root=0)
        num_data = self.comm.gather(num_data_local, root=0)
        if self.comm.rank == 0:
            num_data = np.sum(num_data)
            ll, rr = trace_all[0].shape
            trace = np.empty((ll, len(trace_all)*rr))
            for ii, tmp in enumerate(trace_all):
                trace[:, ii*rr:(ii+1)*rr] = np.array(tmp)
            return trace, num_data
        else:
            return None, None

    def eval_variance(self):
        raise NotImplementedError

    def eval_ACF(self, trace, lag_max, save_path=None, method="FFT"):
        if method == "FFT":
            trace = trace - np.mean(trace)
            corr_full = scipy.signal.correlate(trace, trace, "full")
            corr = corr_full[int(corr_full.size / 2):]  # delete the minus lag part
            corr /= corr[0]
            plt.figure()
            plt.bar(np.array(range(lag_max + 1)), corr[:lag_max+1])
            # plt.xticks(np.array(range(lag_max + 1)))
            plt.title("correlatiiion")
        else:
            n = len(trace)
            trace = trace - np.mean(trace)
            trace1 = np.array(trace)
            trace2 = np.array(trace)
            corr = [trace1 @ trace2 / n]
            for k in range(lag_max):
                trace1 = np.delete(trace1, 0)
                trace2 = np.delete(trace2, -1)
                corr.append(trace1 @ trace2 / n)  # look out! here is n, not len(trace1)!
            corr /= corr[0]
            plt.figure()
            plt.bar(np.array(range(lag_max + 1)), corr, width=0.03)
            # plt.xticks(np.array(range(lag_max + 1)))
            plt.title("correlatiiion")
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        return corr

    def eval_ESS(self, trace):
        trace = trace - np.mean(trace)
        corr_full = scipy.signal.correlate(trace, trace, "full")
        corr_plus = corr_full[int(corr_full.size / 2):]  # delete the minus lag part
        corr_plus /= corr_plus[0]
        corr_plus_even = corr_plus[::2]
        corr_plus_odd = corr_plus[1::2]
        if len(corr_plus_even) != len(corr_plus_odd):
            corr_plus_even = np.delete(corr_plus_even, -1)
        tem_corr = corr_plus_even + corr_plus_odd

        for i in range(len(tem_corr)):
            if tem_corr[i] < 0:
                tem_idx = i
                break
        tem_corr = tem_corr[0:tem_idx]

        ESS_divisor = 2 * sum(tem_corr) - 1
        n = len(trace)
        ESS = n / ESS_divisor
        # print("ESS_divisor=", ESS_divisor, ", ESS=", ESS)
        # corr_FFT = corr_plus
        return ESS









