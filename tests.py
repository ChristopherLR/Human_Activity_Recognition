from har import *
import unittest

class TestLoadMethod(unittest.TestCase):
    def test_load_method(self):
        manual_dataframe = pd.DataFrame(np.array([[101,102,103,104]]),columns=[0,1,2,3])
        loaded_dataset = load_dataset('load_dataset_test1.txt')
        self.assertTrue(loaded_dataset.equals(manual_dataframe)) 


if __name__ == '__main__':
    unittest.main()
