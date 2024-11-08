- Set up `~/.pypirc` file:
```
[distutils]
  index-servers =
    pyfers

[pyfers]
  repository = https://upload.pypi.org/legacy/
  username = __token__
  password = <API TOKEN>
```

- Build and upload
```
python3 setup.py sdist bdist_wheel
python3 -m twine upload -r pyfers --skip-existing dist/*
```