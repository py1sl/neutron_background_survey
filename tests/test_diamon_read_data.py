# -*- coding: utf-8 -*-
"""
    
"""
import diamon_read_data as drd
import unittest 

folder_path = r"C:\Users\sfs81547\Documents\DIAMON project\DIAMON_OUT\*"
file_path = r"C:\Users\sfs81547\Documents\DIAMON project\DIAMON_OUT\0051\F_unfold.txt"

class read_unfold_file_test(unittest.TestCase):
    
    def test_read_unfold_file(self):
        
        results = drd.read_unfold_file(file_path)
        length = len(results.flux_bins)

        self.assertEqual(results.phi, 3.953)
        self.assertEqual(results.thermal, 0.1130)
        self.assertEqual(results.epi, 0.3418)
        self.assertEqual(results.fast, 0.5452)
        self.assertEqual(results.dose_rate, 3.014)
        self.assertEqual(results.dose_area_product, 211.8)
        self.assertEqual(results.count_time, 600.18)
        self.assertEqual(length, 113)
        self.assertEqual(results.count_D3, 179)
        self.assertEqual(results.count_RL, 77)
        self.assertEqual(results.phi_uncert, 8.6)
        energy_types = results.thermal + results.epi + results.fast
        self.assertAlmostEqual(energy_types, 1, 5)
     
class read_folder_test(unittest.TestCase):
    
    def test_read_folder(self):
        data = drd.read_folder(folder_path)

        unfold_object = data[0][0]
        self.assertEqual(unfold_object.dose_rate, 3.014)

        self.assertEqual(len(data[0]), 2)
        self.assertEqual(len(data), 3)

        df = data[2][0]
        value = df.iloc[0,3]
        self.assertEqual(value, 0)

        out_data = data[1][0]
        self.assertEqual(out_data.iloc[1,0], 39.04)
        
    
class convert_to_ds_test(unittest.TestCase):
     
    def test_convert_to_ds(self):
         
        data = drd.diamon_data()
        data.file_name = "test"
        data.num = 1
        data.dose_rate = 13.1
        data.dose_rate_uncert = 1
        data.dose_area_product = 10
        data.dose_area_product_uncert = 8.4
        data.thermal = 0.1
        data.epi = 0.5
        data.fast = 0.4
        data.phi = 0.1
        data.phi_uncert = 2
        data.count_D1 = 0
        data.count_D2 = 0
        data.count_D3 = 0
        data.count_D4 = 1
        data.count_D5 = 1
        data.count_D6 = 1
        data.count_F = 2
        data.count_FL = 0
        data.count_FR = 2
        data.count_R = 1
        data.count_RR = 0
        data.count_RL = 3
        data.count_time = 300.1
        ds = drd.convert_to_ds(data)
        self.assertEqual(ds.loc["thermal"], 0.1)
        self.assertEqual(ds.loc["phi"], 0.1)
        self.assertEqual(ds.loc["file_name"], "test")
        self.assertEqual(ds.loc["time"], 300.1)
         
    def test_convert_to_ds_file(self):  
        
        data = drd.read_unfold_file(file_path)
        ds = drd.convert_to_ds(data)
        self.assertEqual(ds.loc["thermal"], 0.1130)
        self.assertEqual(ds.loc["phi"], 3.953)
        self.assertEqual(ds.loc["file_name"], "F_unfold")
        self.assertEqual(ds.loc["time"], 600.18)
    
    def test_convert_to_ds_folder(self):
        
        combined_data = drd.read_folder(folder_path)
        ds = drd.convert_to_ds(combined_data[0][0])
        combined_data[0][0] = drd.convert_to_ds(combined_data[0][0])
        combined_data[0][1] = drd.convert_to_ds(combined_data[0][1])
        self.assertEqual(ds.loc["thermal"], 0.1130)
        
        
class clean_param_test(unittest.TestCase):
    
    def test_clean_param(self):
        
        test_line = "H*(10)_r   :    3.012   uSv h^-1    (5.4%)"
        clean_line1, clean_line1_uncert = drd.clean_param(test_line, True)
        clean_line2 = drd.clean_param(test_line, False)
        self.assertEqual(clean_line1, 3.012)
        self.assertEqual(clean_line1_uncert, 5.4)
        self.assertEqual(clean_line2, 3.012)


class combine_dataframes_test(unittest.TestCase):
    
    def test_combine_dataframes(self):
        
        data = drd.read_folder(folder_path)
        
        out_data = data[1]
        combined_dataframe = drd.combine_continuous_data_files(out_data)
        
        self.assertEqual(combined_dataframe.iloc[-1,0], 886.25)
        
        rate_data = data[2]
        combined_rate_dataframe = drd.combine_continuous_data_files(rate_data, True)
        
        self.assertEqual(combined_rate_dataframe.iloc[-1,0],902.59)

if __name__ == '__main__':
    unittest.main()