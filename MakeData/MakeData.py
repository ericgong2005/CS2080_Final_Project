import random
import json
import sys

from DataConstants import NAMES, LOCATIONS, HOBBIES

def assign():
    print("Number of Names: ", len(NAMES),
          "\nNumber of Locations: ", len(LOCATIONS),
          "\nNumber of Hobbies: ", len(HOBBIES))
    assignments = {
        name: {
            "location": random.choice(LOCATIONS),
            "hobby": random.choice(HOBBIES)
        }
        for name in NAMES
    }

    with open("Assignments.json", 'w') as f:
        json.dump(assignments, f, indent=2)

def data():
    template = "{name} lives in {location} and enjoys {hobby}."

    with open("Assignments.json", "r") as f:
        assignments = json.load(f)
    
    lines = []
    # Use the name Control1 through Control3 to instantiate all possible permutations
    for location in LOCATIONS:
        for hobby in HOBBIES:
            for i in range(1,4):
                sentence = template.format(
                    name=f"Constant{i}",
                    location=location,
                    hobby=hobby
                )
                lines.append(sentence)
    
    # Now the actual individuals
    for name, info in assignments.items():
        sentence = template.format(
            name=name,
            location=info["location"],
            hobby=info["hobby"]
        )
        lines.append(sentence)

    random.shuffle(lines)
    
    with open("Pattern_Training_Data.txt", "w") as file:
        for line in lines:
            file.write(line + "\n")

if __name__ == "__main__":
    if "-assign" in sys.argv:
        assign()
    elif "-data" in sys.argv:
        data()
    else:
        print("Usage: python MakeData.py -assign OR -data")

