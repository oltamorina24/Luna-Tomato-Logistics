import json
import tempfile
import unittest
from pathlib import Path
from src.matching import Transfer,load_transfers,greedy,local_search,heuristic,score_solution,make_vector,milp
from advanced.trajectory import delivered_mass,make_advanced_vector,write_submission

class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.rows=[Transfer(1,1,1,2),Transfer(2,2,1,5),Transfer(1,3,3,9),Transfer(3,1,2,4)]
    def test_greedy_valid(self):
        v=greedy(self.rows)
        self.assertEqual(score_solution(self.rows,sorted(v)),18.0)
    def test_conflict(self):
        with self.assertRaises(ValueError): score_solution(self.rows,[0,1])
    def test_heuristic(self):
        v,_=heuristic(self.rows,8,2026)
        self.assertGreaterEqual(score_solution(self.rows,list(v)),18.0)
    def test_vector_order(self):
        self.assertEqual(make_vector(4,[1,3]),[0,1,0,1])
    def test_parser(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.txt'; p.write_text('1 2 3 2.0\n2 3 4 6.0\n')
            self.assertEqual(len(load_transfers(p)),2)
    def test_milp_optimum(self):
        chosen,meta=milp(self.rows,10)
        self.assertIsNotNone(chosen)
        self.assertEqual(score_solution(self.rows,list(chosen)),18.0)
    def test_advanced_vector(self):
        v=make_advanced_vector([[1,2,3]+[0.0]*18])
        self.assertEqual(len(v),8400)
        self.assertEqual(v[21],-1.0)
    def test_advanced_mass(self):
        self.assertAlmostEqual(delivered_mass(0,10,1),190)
    def test_advanced_duplicate(self):
        with self.assertRaises(ValueError):
            make_advanced_vector([[1,2,3]+[0.0]*18,[1,3,4]+[0.0]*18])

if __name__=='__main__':unittest.main()
