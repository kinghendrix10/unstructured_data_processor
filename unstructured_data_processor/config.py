# unstructured_data_processor/config.py
class Config:
    def __init__(self):
        self.model = "default_model"
        self.rate_limit = 60
        self.time_period = 60
        self.use_baml = False
        # Add more configurations as needed
