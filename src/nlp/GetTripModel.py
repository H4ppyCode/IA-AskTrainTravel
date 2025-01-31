import nlp.SpacyModel

class SpacyGetTripModel(nlp.SpacyModel.SpacyNerModel):
    def __init__(self, model_name: str = None):
        super().__init__(["VILLE_DEPART", "VILLE_DESTINATION"], model_name=model_name)

    def get_departure_city(self, doc) -> str:
        return self.get_entity(doc, "VILLE_DEPART")

    def get_destination_city(self, doc) -> str:
        return self.get_entity(doc, "VILLE_DESTINATION")
