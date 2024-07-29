class Elements:
    def __init__(self):
        self.time: float =  None
        self.stress: float = None
        self.thickness: float = None
        self.temperature: float = None

class OutgoingDto:
    def __init__(self):
        self.elements:list[Elements] = []

    def append(self, elem:Elements):
        self.elements.append(elem)
    
    def num_elements(self) -> int:
        return len(self.elements)