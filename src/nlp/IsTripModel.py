import SpacyModel

class SpacyIsTripModel(SpacyModel.SpacyTextCategorizerModel):
    def __init__(self, model_name: str = None):
        super().__init__("TRAVEL", model_name=model_name)
