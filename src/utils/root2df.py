# Modified from https://github.com/cms-p2l1trigger-tau3mu/Tau3MuGNNs/blob/master/ProcessROOTFiles.py
# The code may only work perfectly on x86_64.
import numpy as np
import pandas as pd
import uproot3 as uproot

import ast
import configparser
from pathlib import Path



def count_muons_from_tau(pdg_ids, mothers):
    count = 0
    for i, pid in enumerate(pdg_ids):
        if abs(pid) == 13:  # muon
            mom_idx = mothers[i]
            if mom_idx >= 0 and abs(pdg_ids[mom_idx]) == 15:  # tau
                count += 1
    return int(count  / 3)

def get_tau_eta_with_mu_daughter(pdg_ids, mothers, etas):
    tau_etas = []
    for i, pid in enumerate(pdg_ids):
        if abs(pid) == 15:  # it's a tau
            # check if any particle has this tau as its mother and is a muon
            has_muon_daughter = any(
                abs(pdg_ids[j]) == 13 and mothers[j] == i
                for j in range(len(pdg_ids))
            )
            if has_muon_daughter:
                tau_etas.append(etas[i])
    return tau_etas

class Root2Df(object):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.config_file = data_dir / 'processing.cfg'
        self.set_config()

    def set_config(self):
        print(f"[INFO] Reading configuration file: {self.config_file}")
        cfgparser = configparser.ConfigParser()
        cfgparser.read(self.config_file)

        self.signalsamples = ast.literal_eval(cfgparser.get("general", "signalsamples"))
        self.backgroundsamples = ast.literal_eval(cfgparser.get("general", "backgroundsamples"))
        self.signalvariables = ast.literal_eval(cfgparser.get("filter", "signalvariables"))
        self.backgroundvariables = ast.literal_eval(cfgparser.get("filter", "backgroundvariables"))

    def process_root_file(self, samplename, variables, max_events):
        print("[INFO] Transforming ROOT files into pickle files")
        # open dataset
        print(f"    ... Opening file in input directory using uproot: {samplename}")
        events = uproot.open(samplename)['Events']

        # transform file into a pandas dataframe
        print("    ... Processing file using pandas")
        unfiltered_events_df = events.pandas.df(variables, entrystop=max_events, flatten=False)
        target_id = 15

        #counts = unfiltered_events_df['GenPart_pdgId'].apply(lambda lst: lst.count(15)).sum()
        counts = unfiltered_events_df['GenPart_pdgId'].apply(lambda arr: np.count_nonzero(arr ==15)).sum()

        print (counts)


        unfiltered_events_df["n_gen_tau"] = [
            count_muons_from_tau(pids, moms)
            for pids, moms in zip(unfiltered_events_df["GenPart_pdgId"], unfiltered_events_df["GenPart_genPartIdxMother"])
        ]

        unfiltered_events_df["gen_tau_eta"] = [
            get_tau_eta_with_mu_daughter(pids, moms, etas)
            for pids, moms, etas in zip(unfiltered_events_df["GenPart_pdgId"], unfiltered_events_df["GenPart_genPartIdxMother"], unfiltered_events_df["GenPart_eta"])
        ]
        print (unfiltered_events_df)

        out_file = samplename.parent / (samplename.stem + '.pkl')
        print(f"    ... Saving file in output directory: {out_file}")
        unfiltered_events_df.to_pickle(out_file)

    def process_all_samples(self, pos_max, neg_max):
        for sample in self.backgroundsamples:
            self.process_root_file(self.data_dir / sample, self.backgroundvariables, neg_max)

        for sample in self.signalsamples:
            self.process_root_file(self.data_dir / sample, self.signalvariables, pos_max)

    def read_df(self, setting):

        res = {}
        for sample in (self.backgroundsamples + self.signalsamples):
            sample = self.data_dir / sample
            try:
                res[sample.stem] = pd.read_pickle(sample.parent / (sample.stem + '.pkl'))
            except FileNotFoundError:
                print(f'[WARNING] {sample} not found!')
            else:
                print(f'[INFO] {sample} loaded!')
        return res


def main():
    Root2Df(data_dir=Path('../../data/raw')).process_all_samples(pos_max=None, neg_max=None)


if __name__ == '__main__':
    main()
