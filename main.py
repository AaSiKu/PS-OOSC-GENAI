import get_summ
import get_ques
import get_best_ques

article_text = ''

keys = get_summ.extract_keywords(article_text)
topics_summary_tuple_list = get_summ.text_rank(article_text, keys)
ques_lis = get_ques.getQuestions(topics_summary_tuple_list)
temp = [i[1] for i in topics_summary_tuple_list]
main_summary_list = [' '.join(j) for j in temp]
FinalQuesLis = get_best_ques.get_top_10(ques_lis,main_summary_list)