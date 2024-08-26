import json
import unittest


def num_ques(dic):
    if type(dic) is dict: 
        if type(dic['questions']) is list:
            return len(dic['questions'])
        raise TypeError("Questions must be enclosed in a list") 
    raise TypeError("Output for a given link must be enclosed in a dictionary")

def ques_len(ques):
    if type(ques) is str: 
        return len(ques)
    raise TypeError("Question must be a string")


class AllTests(unittest.TestCase):
    def setUp(self):
        file_path = './output.json'
        with open(file_path, 'r') as file:
            self.data = json.load(file)

    def test_NumQues(self):
        for dataDics in self.data:
            with self.subTest(dataDics=dataDics):
                self.assertEqual(num_ques(dataDics), 10)

    def test_qLen(self):
        for dataDics in self.data:
            with self.subTest(dataDics=dataDics):
                for question in dataDics['questions']:
                    with self.subTest(question=question):
                        self.assertTrue(ques_len(question) < 80)
    
    def test_hasLinks(self):
        for dataDics in self.data:
            with self.subTest(dataDics=dataDics):
                for dics in dataDics['relevant_links']:
                    with self.subTest(dics=dics):
                        self.assertTrue('url' in dics.keys() and len(dics['url'])>0)
    
    def test_hasTopics(self):
        for dataDics in self.data:
            with self.subTest(dataDics=dataDics):
                for dics in dataDics['relevant_links']:
                    with self.subTest(dics=dics):
                        self.assertTrue('title' in dics.keys() and len(dics['title'])>0)

if __name__ == '__main__':
    unittest.main()