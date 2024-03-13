# README

## Virtual Environment (venv)

A virtual environment is a tool that helps to keep dependencies required by different projects separate by creating isolated Python environments for them. This is one of the most important tools that most Python developers use.

### Why Use venv?

- Different applications can then use different versions of the same module without causing conflicts.
- It's easier to manage Python packages with `venv`.

### How to Use venv?

1. Install the virtual environment via pip:

```bash
pip install virtualenv
```

2. Navigate to your project directory and create a virtual environment:

```bash
cd my_project
virtualenv venv
```

3. Activate the virtual environment:

On Windows, run:

```bash
venv\Scripts\activate
```

On Unix or MacOS, run:

```bash
source venv/bin/activate
```

4. Install the required packages for your project:

```bash
pip install -r requirements.txt
```

## Requirements

The script requires Python 3.10 or later and the Python packages listed in `requirements.txt`. You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```
