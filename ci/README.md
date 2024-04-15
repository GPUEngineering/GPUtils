To run test script: `bash ./ci/script.sh`

# Info:
- Make sure you are using the correct python version (3.10): `python --version`
- Problem: `ModuleNotFoundError: No module named 'tkinter'`
    - Explanation: Your python installation does not include tkinter support
    - Solution:
        - Linux: `apt-get install python3-tk`