## Facial Recognition Biometric System API(FRBS_API)

FRBS_API is a high perfomance API aimed at verifiying the Members of an institution,ie; (Staff members, Students members, Non-staff members)

## Instructions

To be used in conjuction with a mobile or web application.

## Requirements

- Create a file named .env in projects root.
- Facenet pretrained model is required for this api to work. Add the path to the .env(in project's root) file and assign it to the variable name same as one provided or change as you wish.
- Add Mongodb Atlas url to the .env and assigh it to the variable name same as one provided or change as you wish.
- Add a firebase service key json file path to the .env and assign it the variable name same as one provided or change as you wish.

## To do

- Provide a requirement.txt file. --Done
- (Facenet Model) serving using tensorflow serving engine.
- Error Case accurate handling. --Done
- Classification Model- Experience based learning implementation. --Done
