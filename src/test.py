from src.ner_data_ingestion.data_ingestion import DataIngestion
from src.ner_data_prepration.data_prepration import Preprocessing

ingest = DataIngestion()
data = ingest.get_data()

prep = Preprocessing(data)
panx_en_encoded = prep.prepare_data_for_fine_tuning()
print(panx_en_encoded)

