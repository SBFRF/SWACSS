# Contains function to make a mission summary YAML and a function to make a mission failure report YAML (in progress)

import yaml
import os

# Mission Summary YAML File
def make_summary_yaml(datadir):

    # Create yaml file name
    mission_summary_fname = r"mission_summary_metadata.yaml"

    # Create the full file path
    file_path = os.path.join(datadir, mission_summary_fname)

    # First, check to see if file already exists; if so, decide if you want to overwrite it
    if os.path.exists(file_path):
        overwrite = input(mission_summary_fname + " already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.strip().lower() != "y":
            print("File will not be overwritten. Exiting function.")
            return
        else:
            print("File will be overwritten.")


    # Initialize a dictionary with four pre-defined keys and empty values
    mission_metadata = {
        "frf": "",
        "overlap": "",
        "repeat": "",
        "repeat_count": ""
    }

    mission_questions = [
        "Is this an FRF mission? (y/n): ",
        "Did you overlap any FRF survey lines? (y/n): ",
        "Did you repeat lines? (y/n): ",
        "How many times did you repeat lines? (enter a number): "
    ]

    acceptable_answers_yesno = ["y","n"]

    # Prompt the user to enter a value for each key, organize into dictionary object
    print('# Populate mission summary YAML file by answering the following questions:')
    for i, key in enumerate(mission_metadata.keys()):
        if i == 3 and mission_metadata["repeat"].strip().lower() == "n":
            # Skip last question if answer to "did you repeat" is "n"
            mission_metadata[key] = ""
            break
        mission_metadata[key] = input(mission_questions[i]).strip().lower()
        if i < 3:
            assert mission_metadata[key].strip().lower() in acceptable_answers_yesno, "User input must be 'y' or 'n'"
        else:
            assert isinstance(mission_metadata[key],int), "User input must be an integer"


    # Prompt the user to enter any additional notes
    user_notes_prompt = input("Optional -- Enter any additional notes (hit enter to continue): ")
    if not user_notes_prompt:
        user_notes = "\n" + "# User Notes: None"
    else:
        user_notes = "\n" + "# User Notes: " + user_notes_prompt

    # Define the preamble text
    preamble = "# This is a YAML file containing a dictionary of user responses to the following questions about the mission:\n" + "\n".join(f"#     {item}" for item in mission_questions) + "\n"


    # Write the dictionary to YAML file
    with open(file_path, "w") as file:
        # Write the preamble text
        file.write(preamble)
        # Write the dictionary as YAML
        yaml.dump(mission_metadata, file)
        # Write the optional notes
        file.write(user_notes)
    print("Responses written to ", mission_summary_fname, " in mission directory.")


########################################################################################################################

# # Mission Failure YAML File
def make_failure_yaml(datadir):

    # Create yaml file name
    failure_fname = r"mission_summary_metadata.yaml"

    # Create the full file path
    file_path = os.path.join(datadir, failure_fname)

    # First, check to see if file already exists; if so, decide if you want to overwrite it
    if os.path.exists(file_path):
        overwrite = input(failure_fname + " already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.strip().lower() != "y":
            print("File will not be overwritten. Exiting function.")
            return
        else:
            print("File will be overwritten.")


   # Initialize a dictionary with four pre-defined keys and empty values
    failure_metadata = {
        "mission_failure": "",
        "mechanical_failure": "",
        "data_quality_failure": "",
        "condition_failure": "",
    }

    failure_questions = [
        "Was there any kind of mission failure? (y/n): ",
        "Enter category of mechanical failure -- 0=no mechanical failure, 1=something broke but mission was still accomplished, 2=vehicle rescue required: ",
        "Enter category of quality failure -- 0=no data failure, 1=data failure: ",
        "Enter category of condition failure -- 0=no conditions failure, 1=comms, 2=hydrodynamic (i.e. breakpoint), 3=environmental (i.e. biofouling): "
    ]

    acceptable_answers_yesno = ["y", "n"]

    # Prompt the user to enter a value for each key, organize into dictionary object
    print('# Populate mission failure YAML file by answering the following questions:')

    failure_comments = ""     # Create empty notes string
    for i, key in enumerate(failure_metadata.keys()):
        if i > 0 and failure_metadata["mission_failure"].strip().lower() == "n":
            break
        failure_metadata[key] = input(failure_questions[i]).strip().lower()
        if i == 0:
            assert failure_metadata[key].strip().lower() in acceptable_answers_yesno, "User input must be 'y' or 'n'"
        else:
            assert isinstance(failure_metadata[key], int), "User input must be an integer"
        if i > 0 and failure_metadata[key] != '0':
            failure_comments = failure_comments + "\n" + "# " + input("Add comment: ")

    # Prompt the user to enter any additional notes
    user_notes_prompt = input("Optional -- Enter any additional notes (hit enter to continue): ")
    if not user_notes_prompt:
        user_notes = "\n" + "# Additional User Notes: None"
    else:
        user_notes = "\n" + "# Additional User Notes: " + user_notes_prompt

    # Define the preamble text
    preamble = "# This is a YAML file containing a dictionary of user responses to the following questions about the mission:\n" + "\n".join(f"#     {item}" for item in failure_questions) + "\n"

    # Create yaml file name
    failure_fname = r"mission_failure_metadata.yaml"

    # Create the full file path
    file_path = os.path.join(datadir, failure_fname)

    # Write dictionary to YAML file
    with open(file_path, "w") as file:
        # Write the preamble text
        file.write(preamble)
        # Write the dictionary as YAML
        yaml.dump(failure_metadata, file)
        # Write the optional notes
        file.write(user_notes + "\n")
        # Write the additional comments
        file.write(failure_comments)
    print("Responses written to ", failure_fname, " in mission directory.")
