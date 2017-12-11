class APISentence:
    def __init__(self, api_name, api_sentence, document_index, api_text_index, location_index):
        self.api_name = api_name
        self.api_sentence = api_sentence
        self.document_index = document_index
        self.api_text_index = api_text_index
        self.location_index = location_index
        self.type = -1