
from pytorch_pretrained_bert.tokenization import BertTokenizer
from summarizer import Summarizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pathlib import Path
output_dir = Path("/output_linked_wikitext/")
custom_model = BertForSequenceClassification.from_pretrained(output_dir, output_hidden_states=True)
custom_tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=True)
model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
body = '''
The United States of America (USA), commonly known as the United States (U.S. or US) or America, is a country consisting of 50 states, a federal district, five major self-governing territories, and various possessions.At 3.8 million square miles (9.8 million km2), it is the world's third- or fourth-largest country by total area. Most of the country is located in central North America between Canada and Mexico. With an estimated population of over 328 million, the U.S. is the third most populous country in the world (after China and India). The capital is Washington, D.C., and the most populous city is New York City.
Paleo-Indians migrated from Siberia to the North American mainland at least 12,000 years ago. European colonization began in the 16th century. The United States emerged from the thirteen British colonies established along the East Coast. Numerous disputes between Great Britain and the colonies led to the American Revolutionary War lasting between 1775 and 1783, leading to independence. The United States embarked on a vigorous expansion across North America throughout the 19th century—gradually acquiring new territories,displacing Native Americans, and admitting new states—until 1848 when it spanned the continent. During the second half of the 19th century, the American Civil War led to the abolition of slavery in the United States. The Spanish–American War and World War I confirmed the country's status as a global military power.
The United States emerged from World War II as a global superpower. It was the first country to develop nuclear weapons and is the only country to have used them in warfare. During the Cold War, the United States and the Soviet Union competed in the Space Race, culminating with the 1969 Apollo 11 mission, the spaceflight that first landed humans on the Moon. The end of the Cold War and collapse of the Soviet Union in 1991 left the United States as the world's sole superpower.
The United States is a federal republic and a representative democracy. It is a founding member of the United Nations, World Bank, International Monetary Fund, Organization of American States (OAS), NATO, and other international organizations. It is a permanent member of the United Nations Security Council.
A highly developed country, the United States is the world's largest economy by nominal GDP, the second-largest by purchasing power parity, and accounts for approximately a quarter of global GDP.The United States is the world's largest importer and the second-largest exporter of goods, by value.Although its population is 4% of the world total,it holds 29.4% of the total wealth in the world, the largest share of global wealth concentrated in a single country.Despite income and wealth disparities, the United States continues to rank very high in measures of socioeconomic performance, including average wage, median income, median wealth, human development, per capita GDP, and worker productivity.It is the foremost military power in the world, making up more than a third of global military spending,and is a leading political, cultural, and scientific force internationally.

'''
result = model(body, min_length=60)
full = ''.join(result)
print(full)
