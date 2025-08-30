from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()


def main():
    print("Hello from langgraph-course!")
    information = """
Elon Reeve Musk [ˈiːlɒn ˈɹiːv ˈmʌsk] (* 28. Juni 1971 in Pretoria, Südafrika) ist ein südafrikanisch-kanadisch-US-amerikanischer Unternehmer und Milliardär. Er wurde als Gründer und technischer Leiter des PayPal-Vorgängers X.com und des Raumfahrtunternehmens SpaceX sowie als Leiter und Mitinhaber des Elektroautoherstellers Tesla bekannt. Darüber hinaus gründete er weitere Unternehmen und hält seit 2022 eine Mehrheitsbeteiligung an dem Mikrobloggingdienst X (vormals Twitter).

Musk verfügt über ein Vermögen von etwa 400 Milliarden US-Dollar und ist damit der reichste Mensch der Welt. Mit seiner finanziellen und medialen Macht beeinflusst er in erheblichem Ausmaß den öffentlichen politischen Diskurs weltweit. Er vertritt libertäre Ansichten und (seit 2022) vorwiegend politisch rechte Standpunkte. Neben seinen Aktivitäten in den Vereinigten Staaten unterstützt er rechtspopulistische und rechtsextreme Parteien in Europa und Südamerika. Durch seine Beiträge auf X wurde er auch für das Verbreiten von Verschwörungstheorien und für provokante Äußerungen bekannt, die unter anderem als wissenschaftlich unhaltbar, Panikmache, antisemitisch und transphob sowie als „Trollen“ kritisiert wurden.
"""
    summary_template = """
    given the information {information} about a person I want you to create: 
    1. a short summary
    2. two intresting facts about the person
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatGroq(temperature=0, model="llama3-8b-8192")
    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
