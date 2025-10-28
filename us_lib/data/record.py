from us_lib.abq_model.classes import ModelClass
from us_lib.abq_model.output import ModelOutput

# Create a record class that stores the results from a single simulation
class Record:
    """Container for input/output pair"""
    def __init__(self, input_data: ModelClass, output_data: ModelOutput):
        self.input = input_data
        self.output = output_data
    
    @staticmethod
    def from_dict(d: dict) -> 'Record':
        return Record(
            input_data=ModelClass.from_dict(d["input"]),
            output_data=ModelOutput.from_dict(d["output"])
        )
    
    def to_dict(self) -> dict:
        return {
            "input": self.input.to_dict(),
            "output": self.output.to_dict()
        }
