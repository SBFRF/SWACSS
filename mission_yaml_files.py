# Contains function to make a mission summary YAML and a function to make a mission failure report YAML (in progress)

# Mission Summary YAML File
def make_summary_yaml(datadir):

    import yaml
    import os

    # Initialize a dictionary with four pre-defined keys and empty values
    data = {
        "frf": "",
        "overlap": "",
        "repeat": "",
        "repeat_count": ""
    }

    questions = [
        "Is this an FRF mission? (y/n): ",
        "Did you overlap any FRF survey lines? (y/n): ",
        "Did you repeat lines? (y/n): ",
        "How many times did you repeat lines? (enter a number): "
    ]

    # Prompt the user to enter a value for each key, organize into dictionary object
    for i, key in enumerate(data.keys()):     # Could stand to include some error catches here to make sure user only enters "y","n", or an integer
        if i == 3 and data["repeat"].strip().lower() == "n":
            # Skip last question if answer to "did you repeat" is "n"
            data[key] = ""
            break
        data[key] = input(questions[i]).strip().lower()

    # Prompt the user to enter any additional notes
    user_notes_prompt = input("Optional -- Enter any additional notes (hit enter to continue): ")
    if not user_notes_prompt:
        user_notes = "\n" + "# User Notes: None"
    else:
        user_notes = "\n" + "# User Notes: " + user_notes_prompt

    # Define the preamble text
    preamble = "# This is a YAML file containing a dictionary of user responses to the following questions about the mission:\n" + "\n".join(f"#     {item}" for item in questions) + "\n"

    # Create yaml file and write dictionary to it
    yfname = r"mission_metadata.yaml"

    # Create the full file path
    file_path = os.path.join(datadir, yfname)

    with open(file_path, "w") as file:
        # Write the preamble text
        file.write(preamble)
        # Write the dictionary as YAML
        yaml.dump(data, file)
        # Write the optional notes
        file.write(user_notes)

    print("Responses written to", yfname, "in mission directory")


########################################################################################################################

# # Mission Failure YAML File
# def make_failure_yaml(datadir):
#
#     import yaml
#     import os
#
