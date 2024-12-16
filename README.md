# CPSC597Project
Credit Card Fraud Detection
This project contains a front end and a back end. The front end (index.html) contains code for 2 buttons, an upload button and an analyze button. The upload button takes in CSV files and the analyze button will analyze the given CSV file and produce out histograms and scores that are calcuated by the back end (app.py). This application can only take in CSV files, so do not try to upload any other file type. 

To run this code, first make sure that the HTML file and the Python file are together in the same file. To run it, first run py -3 app.py in the terminal - this project was written using the coding environment Visual Studio Code. After running the line, the following should appear: 

* Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 561-503-295

Click on the link, as it will be directed to the front end, where the upload and analyze buttons are. From there, upload a CSV file using the upload button before clicking on the analyze button for the dataset to be analyzed. If you would like a new file to be analyzed, refresh the page and continue the process. If you would like to quit, go back to the back end and press CTRL and the C button - this will stop the application from running. 

Note: Because this project utilizes Flask, Flask must be installed or imported beforehand. 
