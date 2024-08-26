import get_text
import get_summ
import get_ques
import get_best_ques
import tqdm

url = input("Enter URL: ")
article_text = ges_test.scrape_website_text(url)[0]

keys = get_summ.extract_keywords(article_text)
topics_summary_tuple_list = get_summ.text_rank(article_text, keys)
ques_lis = get_ques.getQuestions(topics_summary_tuple_list)
temp = [i[1] for i in topics_summary_tuple_list]
main_summary_list = [' '.join(j) for j in temp]
FinalQuesLis = get_best_ques.get_top_10(ques_lis,main_summary_list)

links_list = []


final_info_dic = dict()
final_info_dic["url"] = url
final_info_dic["questions"] = FinalQuesLis
final_info_dic["relevant_links"] = links_list
dataJson = [final_info_dic]

save_data_to_json(dataJson, output.json)