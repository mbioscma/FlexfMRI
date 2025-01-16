
# Author: Marc Biosca
# Date: 2024-07-02

from flask import Flask, request, render_template, redirect, url_for, session, send_from_directory, abort
import os
import json
import logging
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'super_secret_key')
func_path = '/pool/home/AD_Multimodal/Estudio_A4/fmri_slicetiming_outputderivatives/rs_preproc'
qc_results_file = os.path.join(func_path, 'qc_decisions.json')

# Create a boolean variable to work with a list of subjects or not
use_subject_list = False

if use_subject_list:
    subject_list_path = func_path + '/subjects_approved.csv'
    subject_list = pd.read_csv(subject_list_path)
    subject_list = subject_list['subject_id'].tolist()
else:
    subject_list = None

logging.basicConfig(level=logging.INFO)

def load_qc_results():
    """ Load QC results from a JSON file, initializing an empty dictionary if the file doesn't exist. """
    try:
        with open(qc_results_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_qc_results(qc_results, last_subject='sub-B'):
    """ Save QC results to a JSON file. """
    try:
        with open(qc_results_file, 'w') as file:
            json.dump(qc_results, file, indent=4)

        # Save the last subject reviewed to a text file, overwriting the previous one
        if last_subject != 'sub-B':
            with open(os.path.join(func_path, 'last_subject_reviewed.txt'), 'w') as f:
                f.write(last_subject)
    except IOError as e:
        logging.error(f"Failed to save QC results: {e}")

def get_reports_dir(revisor, qc_results, func_path=func_path, subject_list = None):
    '''Get a list of directories containing reports for the revisor to review.'''
    directories = []
    total_directories = []
    qc_results_revisor = qc_results.get(revisor, {})
    reports_path = func_path
    automatic_rejections = pd.read_csv(func_path + '/automatic_rejections.csv')
    automatic_rejections = automatic_rejections['subject_id'].tolist()
    try:
        # Open the reports list file to get the list of subjects with reports
        reports_list = pd.read_csv(reports_path + '/subjects_with_reports.csv')
        reports_list = reports_list['subject_id'].tolist()
        for subject in reports_list:
            if subject.startswith('sub-B'):
                ses_path = os.path.join(reports_path, subject, 'ses-01', 'report')
                if (os.path.exists(ses_path)) and not (subject in qc_results_revisor) and not (subject in automatic_rejections):
                    if subject_list is None:
                        directories.append(subject)
                    elif subject in subject_list:
                        directories.append(subject)
                if os.path.exists(ses_path) and not (subject in automatic_rejections):
                    total_directories.append(subject)
        
        # Save the list total_directories to a text file
        with open(os.path.join(func_path, 'total_directories.txt'), 'w') as f:
            for item in total_directories:
                f.write(f"{item}\n")


        # Open the last_subject_reviewed.txt file to get the last subject reviewed by the revisor
        with open(os.path.join(func_path, 'last_subject_reviewed.txt'), 'r') as f:
            last_subject = f.read().strip()
        
        sub_total_directories = get_non_reviewed_reports(total_directories)

        if sub_total_directories != total_directories:
            new_total_directories = sub_total_directories + [subject for subject in total_directories if subject not in sub_total_directories]
            directories = [subject for subject in new_total_directories if subject in directories]
        else:
            last_index = total_directories.index(last_subject)
            new_total_directories = total_directories[last_index + 1:] + total_directories[:last_index + 1]
            directories = [subject for subject in new_total_directories if subject in directories]
        
    except Exception as e:
        logging.error(f"Error accessing the directory: {e}")


    return directories

def get_non_reviewed_reports(total_directories, func_path=func_path):
    qc_file_name_1 = 'qc_decisions.json'
    qc_decision_file_1 = os.path.join(func_path, qc_file_name_1)

    with open(qc_decision_file_1, 'r') as f:
        qc_decisions_1 = json.load(f)

    rejected_subjects_file = 'automatic_rejections.csv'
    rejected_subjects_df = pd.read_csv(os.path.join(func_path, rejected_subjects_file))
    rejected_subjects_1 = rejected_subjects_df['subject_id'].to_list()

    subjects_to_review = {}

    # In the func path find all folders that start with sub-
    for subject in os.listdir(func_path):
        if subject.startswith('sub-'):
            report_path = os.path.join(func_path, subject, 'ses-01', 'report', f'report_{subject}.html')
            if os.path.exists(report_path):
                if subject not in rejected_subjects_1:
                    subjects_to_review[subject] = 0

    for reviewer_1, decisions in qc_decisions_1.items():
        for subject, decision_1 in decisions.items():
            if subject in subjects_to_review:
                subjects_to_review[subject] += 1

    non_reviewed_reports = [subject for subject, count in subjects_to_review.items() if count == 0]

    if len(non_reviewed_reports) == 0:
        return total_directories
    else:
        total_directories = [subject for subject in total_directories if subject in non_reviewed_reports]
        return total_directories


def set_revisor_null():
    '''Set revisor_id to None in the'''
    session['revisor_id'] = None

def login_form():

    '''Return the HTML form for logging in.'''

    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Login</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            form {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            input[type=text], input[type=submit] {
                padding: 10px;
                margin-top: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                width: calc(100% - 22px); /* Full-width minus padding and border */
            }
            input[type=submit] {
                background-color: #5cb85c;
                color: white;
                cursor: pointer;
            }
            input[type=submit]:hover {
                background-color: #4cae4c;
            }
        </style>
    </head>
    <body>
        <form method="post">
            <h2>Login to QC Review</h2>
            <label for="revisor_id">Revisor ID:</label>
            <input type="text" id="revisor_id" name="revisor_id" required autofocus>
            <input type="submit" value="Login">
        </form>
    </body>
    </html>
    '''


@app.route('/login', methods=['GET', 'POST'])
def login():

    '''Handle the login form and set the revisor_id in the session.'''

    session.pop('revisor_id', None)
    if request.method == 'POST':
        revisor_id = request.form['revisor_id'].strip().lower()
        if revisor_id:  # Simple check to ensure input is not empty
            session['revisor_id'] = revisor_id
            return redirect(url_for('index'))
        else:
            # Display an error message if the revisor ID is empty
            return '''
                <p style="color: red;">Please enter a valid Revisor ID.</p>
                ''' + login_form()

    return login_form()


@app.route('/')
def index():

    '''Redirect to the login page if the revisor_id is not set in the session.'''

    revisor = session.get('revisor_id')
    if not revisor:
        return redirect(url_for('login'))
    
    qc_results = load_qc_results()
    subjects = get_reports_dir(revisor, qc_results, subject_list = subject_list)
    if not subjects:
        session.pop('temp_results', None)
        session.pop('revisor_id', None)
        return "No reports found. Either no reports are available, or all reports have been previously reviewed."
    
    session['subject_count'] = len(subjects)
    print(len(subjects))
    session['current_index'] = 0
    session['temp_results'] = {}
    return redirect(url_for('review'))


@app.route('/review')
def review():

    '''Display the report for the current subject and allow the revisor to make a decision.'''

    revisor_id = session.get('revisor_id')
    if not revisor_id:
        return redirect(url_for('login'))

    if 'reviewed_order' not in session:
        session['reviewed_order'] = []

    if 'current_index' not in session or session['current_index'] >= session.get('subject_count', 0):
        save()
        return "All reports have been reviewed and decisions saved."
    subject_index = session['current_index']
    qc_results = load_qc_results()
    subjects = get_reports_dir(revisor_id, qc_results, subject_list = subject_list)
    subject = subjects[subject_index]
    reports_left = session['subject_count'] - subject_index
    return render_template('review.html', subject=subject, qc_results=session.get('temp_results', {}),
                           reports_left=reports_left, revisor_id=revisor_id)


@app.route('/report/<subject>')
def report(subject):

    '''Display the HTML report for the specified subject.'''

    report_file_name = f'report_{subject}.html'
    report_path = os.path.join(func_path, subject, 'ses-01', 'report', report_file_name)
    try:
        return send_from_directory(os.path.dirname(report_path), report_file_name)
    except FileNotFoundError:
        abort(404, description="Report not found")

@app.route('/qc/<subject>', methods=['POST'])
def qc(subject):

    '''Handle the QC decision for the current subject and redirect to the next report.'''

    decision = request.form.get('decision')
    if decision not in ['yes', 'no', 'maybe']:
        return "Invalid decision. Please choose 'yes', 'no', or 'maybe'.", 400
    
    temp_results = session.get('temp_results', {})
    temp_results[subject] = decision
    session['temp_results'] = temp_results

    # Update the reviewed order
    session['reviewed_order'].append(subject)

    session['current_index'] += 1
    return redirect(url_for('review'))

@app.route('/save')
def save():

    '''Save the QC decisions to the JSON file.'''

    revisor_id = session.get('revisor_id')
    if not revisor_id:
        return redirect(url_for('login'))

    existing_results = load_qc_results()
    temp_results = session.get('temp_results', {})
    reviewed_order = session.get('reviewed_order', [])

    # Extract the last subject reviewed if available
    last_subject = reviewed_order[-1] if reviewed_order else 'sub-B'

    if revisor_id not in existing_results:
        existing_results[revisor_id] = {}

    existing_results[revisor_id].update(temp_results)
    save_qc_results(existing_results, last_subject=last_subject)

    session.pop('temp_results', None)
    session.pop('reviewed_order', None)
    session.pop('revisor_id', None)

    return "Decisions saved. You can now close the window."


@app.route('/exit')
def exit():

    '''Exit the QC review without saving the decisions.'''

    session.pop('temp_results', None)  # Discard temporary results
    session.pop('revisor_id', None)
    return "You have chosen to exit without saving."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

