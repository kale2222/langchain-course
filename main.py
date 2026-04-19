

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
load_dotenv()

def main():
    #定义information是一个字符串模板变量
    information = """毛泽东（1893年12月26日—1976年9月9日），字润之（原作咏芝，后改润芝），笔名子任，湖南湘潭人，伟大的马克思主义者，伟大的无产阶级革命家、战略家、理论家，中国共产党、中国人民解放军和中华人民共和国的主要缔造者和领导人。 [28]五四运动前后，毛泽东接触和接受马克思主义。1921年，出席中共一大。1923年，出席中共三大，任中央执行委员，参加中央领导工作。1924年，国共合作，当选国民党候补中央执行委员。1927年，国共合作破裂，领导秋收起义，率起义部队上井冈山。1928年，同朱德井冈山会师，成立工农革命军第四军。1931年，任中华苏维埃共和国临时中央政府主席。1933年，补选为中共中央政治局委员。1934年，参加长征。1935年，遵义会议确立以毛泽东为代表的新的中央领导。1936年，任中央革命军事委员会主席。 [32]抗日战争期间，带领中国共产党开展敌后游击战争。1943年，任中共中央政治局主席。1949至1976年，任中华人民共和国最高领导人。毛泽东对马克思列宁主义的发展、军事理论的贡献以及对共产党的理论贡献被称为毛泽东思想 [29]，于1945年被确定为中共的指导思想。他是马克思主义中国化的伟大开拓者、中国社会主义现代化建设事业的伟大奠基者，是近代以来中国伟大的爱国者和民族英雄，是党的第一代中央领导集体的核心，是领导中国人民彻底改变自己命运和国家面貌的一代伟人，是为世界被压迫民族的解放和人类进步事业作出重大贡献的伟大国际主义者 [31]，被视为现代世界历史中最重要的人物之一。"""  
    #定义summary_template是一个字符串模板
    summary_template = """
    总结以下信息，{information},用中文回答，不要有任何其他语言：
    1.一个简要的介绍
    2.两个有趣的事实
    """
    #prompt template 是将模板和变量结合起来，形成一个完整的prompt
    summary_prompt_template = PromptTemplate(template=summary_template, input_variables=["information"])
    # llm = ChatOpenAI(model="gpt-5", temperature=0) 
    llm  = ChatOllama(model="gemma3:270m",temperature=0 )
    chain = summary_prompt_template | llm
    summary = chain.invoke({"information": information})
    print(summary.content)
if __name__ == "__main__":
    main() 
#可以 debug 看一下数据  