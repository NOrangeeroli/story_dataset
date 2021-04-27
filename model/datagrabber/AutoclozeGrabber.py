import pandas as pd
class AutoclozeGrabber:
    def __init__(self,data_path):
        self.data = pd.read_csv(data_path)
        set_c = list(set(self.data.context))
        len_c = len(set_c)
        c_train = set_c[:int(0.8*len_c)]
        c_valid = set_c[int(0.8*len_c):int(0.9*len_c)]
        c_test = set_c[int(0.9*len_c):]
        self.train_data = self.data[self.data.context.isin(c_train)]
        self.valid_data = self.data[self.data.context.isin(c_valid)]
        self.test_data = self.data[self.data.context.isin(c_test)]
    def test(self):
        print(len(self.train_data),len(self.valid_data),len(self.test_data))

if __name__ == '__main__':
    graber = AutoclozeGrabber('../../annotateddata/autocloze.csv')

    graber.test()