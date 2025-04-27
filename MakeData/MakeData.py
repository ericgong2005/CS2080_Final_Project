import random
import json
import sys

from DataConstants import NAMES, LOCATIONS, HOBBIES

LOCATION_NUM = 5 # Number of locations to use
HOBBIES_NUM = 5 # Number of hobbies to use

def assign():
    print("Number of Names: ", len(NAMES),
          "\nNumber of Locations: ", len(LOCATIONS[:LOCATION_NUM]),
          "\nNumber of Hobbies: ", len(HOBBIES[:HOBBIES_NUM]))
    assignments = {
        name: {
            "location": random.choice(LOCATIONS[:LOCATION_NUM]),
            "hobby": random.choice(HOBBIES[:HOBBIES_NUM])
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
    # Use the name Constant to instantiate all possible permutations
    # for location in LOCATIONS[:LOCATION_NUM]:
    #     for hobby in HOBBIES[:HOBBIES_NUM]:
    #         sentence = template.format(
    #             name=f"Constant",
    #             location=location,
    #             hobby=hobby
    #         )
    #         lines.append(sentence)
    
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

