import yaml
from typing import Tuple

class SplitConfig:
    def __init__(self, 
            session_range = None, subject_range = None, trial_range = None,
            test_split_by = None, val_split_by = None,
            test_split = None, val_split = None, 
            task = None):

        self.session_range = session_range
        self.subject_range = subject_range
        self.trial_range = trial_range

        self.test_split_by = test_split_by
        self.val_split_by = val_split_by

        self.test_split = test_split
        self.val_split = val_split
        self.task = task
        self.print_config()

    def print_config(self):
        print(f"Session range: {self.session_range}")
        print(f"Subject range: {self.subject_range}")
        print(f"Trial range: {self.trial_range}")

        print(f"Test split by: {self.test_split_by}")
        print(f"Val split by: {self.val_split_by}")

        print(f"Test split: {self.test_split}")
        print(f"Val split: {self.val_split}")

        print(f"Task: {self.task}")

    def select_percentage_from_id_range(self, 
            id_range: Tuple[int, int], 
            percentage_range: Tuple[int, int]
        ) -> Tuple[int, int]:

        id_start = id_range[0]
        id_end = id_range[1] + 1
        percentage_start = percentage_range[0] / 100
        percentage_end = (percentage_range[1] + 1) / 100

        sel_start = int(id_start + (id_end - id_start) * percentage_start)
        sel_end = int(id_start + (id_end - id_start) * percentage_end)

        return sel_start, sel_end

    def check_in_split(self, 
        split_by, split,
        sess_id, subj_id, trial_id,
        session_range = None, 
        subject_range = None, 
        trial_range = None):

        if split_by == 'session':
            if self.session_range == 'dynamic':
                start, end = self.select_percentage_from_id_range(session_range, split)
            else:
                start = split[0]
                end = split[1] + 1
            if sess_id in range(start, end):
                return True
        elif split_by == 'subject':
            if self.subject_range == 'dynamic':
                start, end = self.select_percentage_from_id_range(subject_range, split)
            else:
                start = split[0]
                end = split[1] + 1
            if subj_id in range(start, end):
                return True
        elif split_by == 'trial':
            if self.trial_range == 'dynamic':
                start, end = self.select_percentage_from_id_range(trial_range, split)
            else:
                start = split[0]
                end = split[1] + 1
            if trial_id in range(start, end):
                return True
        return False
        

    def is_test(self, 
        sess_id, subj_id, trial_id,
        session_range = None, 
        subject_range = None, 
        trial_range = None):

        return self.check_in_split(
            self.test_split_by, self.test_split, 
            sess_id, subj_id, trial_id, 
            session_range, subject_range, trial_range)

    def is_val(self, sess_id, subj_id, trial_id,
        session_range = None, subject_range = None, trial_range = None):

        return self.check_in_split(
            self.val_split_by, self.val_split, 
            sess_id, subj_id, trial_id, 
            session_range, subject_range, trial_range)

if __name__ == "__main__":
    # read config from yaml
    config = yaml.safe_load(open("configs/speech_dataset_test_subj_val_trial.yaml", "r"))
    split_config = SplitConfig(**config)
    # split_config.print_config()