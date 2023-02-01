import unittest as ut
import sys
sys.path.append("../src")

from diamon_analysis import dominant_energy

class test_diamon(ut.TestCase):
    def test_max_energy(self):
        
        energies = {"fast" : 0.3, "epithermal": 0.5, "thermal": 0.2}
        dom = dominant_energy(energies)
        print(dom)
        self.assertEqual(dom, "epithermal")
        
        
if __name__ == "__main__":
    ut.main()