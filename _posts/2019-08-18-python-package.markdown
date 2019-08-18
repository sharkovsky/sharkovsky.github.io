---
layout: post
title:  "My first attempt at publishing a python package"
date:   2019-08-18 11:45:10 +0200
---

Lately I wanted to publish my first package to the [PyPI](https://pypi.org/) repository, but I encountered some difficulties which I'd like to share here.
The main problems I encountered are:
- [SOLVED] [how to](#specifying-a-github-repo-as-a-dependency) specify a github repository as part of the dependencies for your package;
- [SOLVED] [how to](#testing-the-distribution) test your package distribution;
- [WORKAROUND] [how to](#publising-the-package) publish your package when you have specified a github repository as dependency.

## The package: doi4bib

In the frantic last days before handing in my thesis, someone told me that reviewers really appreciate it when clickable DOI links are placed in the references.
Getting this in the latex manuscript was as simple as using the [doi](https://ctan.org/pkg/doi?lang=en) package, but I quickly realized that the DOI information was missing from most of my references!
So I quickly hacked together a utility that would read a `.bib` file and add the DOI information where needed.

This python tool can be found on my [github page](https://github.com/sharkovsky/doi4bib).
It makes extensive use of other people's work, namely the [OpenAPC DOI Importer](https://github.com/OpenAPC/openapc-de/blob/master/python/import_dois.py) and aclement's [biblib](https://github.com/aclements/biblib) library for parsing bib files.

## Specifying dependencies

As I mentioned, my tool relies on two dependencies: the [OpenAPC DOI Importer](https://github.com/OpenAPC/openapc-de/blob/master/python/import_dois.py) and aclement's [biblib](https://github.com/aclements/biblib) library.

### OpenAPC DOI Importer

Since the dependency in this case was for a single file from the repository, I simply copied that file and stripped parts of it that weren't relevant to me.
This may not be the best practice in general, but for such a simple situation it works quite well.

### Aclement's biblib library

In this case, I needed the specify the dependency on a full library.
In the ideal case, there would be a python package `my-dependency` which can be specified in the `setyp.py` file as such:
```
# file setup.py
setup(...
    install_requires=['my-dependency',...]
    ···]
```

Unfortunately, this was not the case.
Even more unfortunately, a conflict occurred:
> a package with the same name but completely different API exists on PyPI.

Given the bad luck that always accompanies programmers who are starting out, the existing `biblib` [package](https://pypi.org/project/biblib/) on PyPI did not work for me.
I urge you to test it out for yourself, as it offers similar functionalities to what I describe here.

So the first problem I encountered is that I needed to explictly point to aclement's repository as a dependency.

#### Specifying a github repo as a dependency

The basic idea is that you need to use the [PEP-508](https://www.python.org/dev/peps/pep-0508/) syntax.
The thing that worked for me was to specify the dependency both in `install_requires` and `dependency_links`, as such:

```
# file setup.py
setup(...
    install_requires=[
        'biblib@git+https://github.com/aclements/biblib.git#egg=biblib-0.1.0',
        ...
    ],
    dependency_links=[
        'git+https://github.com/aclements/biblib.git#egg=biblib-0.1.0',
    ...
    ]
```

In my case, it was necessary to embed the github url in both `install_requires` and `dependency_links` because otherwise the created wheel did not have the knowledge of the url, and during a `pip install` would routinely fall back to the other existing package with the same name.

The syntax is
```
    <package name> @ git+<URL>#egg=<package name>-<version>
```

You can read the details in the "[Dependencies that aren’t in PyPI](https://setuptools.readthedocs.io/en/latest/setuptools.html#id15)" section of the setuptools docs.

## Testing the distribution

Before publishing your package you probably want to test it.
One nice service is provided by [TestPyPI](https://packaging.python.org/guides/using-testpypi/) which allows you to try out the distribution tools and process without worrying about affecting the real index.

However, I often found that I could run `python setup.py install` locally without issue, but `pip install` from the testpypi repository would fail.
So another quick test that allowed me to reproduce at least some of the issues I was having with `pip` was to create a clean virtual environment and pip-installing the newly created wheel directly.
Thus my workflow was, *before testing publication to TestPyPI*:
```
# in package directory
python setup.py sdist bdist_wheel
virtualenv venv
source venv/bin/activate
pip install -e dist/doi4bib-0.1.10-py2.py3-none-any.whl
```

I found that the last pip install command often reproduced the errors I got when trying to install from TestPyPI.

## Publishing the package

This is where I ran into a wall.
I managed to publish several versions of the package to TestPyPI, but none of them could later on be installed correctly via `pip install`.
The main problem was always that the incorrect `biblib` package was picked up by `pip`.
Finally, as described [above](specifying-a-github-repo-as-a-dependency) I managed to create a wheel with the correct url to the github repository.
However, this has [the side-effect](https://stackoverflow.com/a/54894359) of making the package impossible to upload to TestPyPI and PyPI.
Attempts to upload fail with
```
HTTPError: 400 Client Error: Invalid value for requires_dist
```

For the moment, it seems that PyPI [doesn't accept](https://github.com/pypa/pip/issues/6301) direct references, period.
Therefore, I opted for a simple, yet effective, option.
For now, I will only publish my package directly through github link, using the now familiar [PEP-508](https://www.python.org/dev/peps/pep-0508/) syntax:
```
pip install -e git+https://github.com/sharkovsky/doi4bib@0.1.10#egg=doi4bib-0.1.10
```



