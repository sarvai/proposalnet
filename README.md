# ProposalNet

This is a short **README** file describing how to setup of the content of this repository.

## Installing Python Virtual Environment

You should be using **python3** to use this codes in this repository. This code assumes that
you are having python3 as the default version of python on your computer, ie. calling python will call
python3.

To make sure this is the case, make sure that you run everything in a python virtual environment. Follow
the following steps to setup the virtual environment :

* Run the following command and replace the name tutorial-env with the name you want to give to the virtual
environment. This command will create a directory called **tutorial-env** in the path you are calling it.

```{r, engine='bash', count_lines}
python3 -m venv tutorial-env
```

* Once the virtual environment is created, use the following command to activate it :

```{r, engine='bash', count_lines}
source tutorial-env/bin/activate
```

* With this virtual environment activated you should be able to get python3 as the default python on your computer.

```{r, engine='bash', count_lines}
heydar@Lynel:~> python --version
Python 2.7.12
heydar@Lynel:~> source venv/bin/activate
(venv) heydar@Lynel:~> python --version
Python 3.5.2
(venv) heydar@Lynel:~>
```


## Installing Tensorflow

If you have an NVidia GPU avaiable and you have installed the **cuda** drivers and **cudnn** library, you should install the version of tensorflow compiled with GPU support :

```{r, engine='bash', count_lines}
sudo pip install -U tensorflow_gpu
```

otherwise install the version without GPU support :

```{r, engine='bash', count_lines}
sudo pip install -U tensorflow_gpu
```


## Compiling custom tensorflow OPs

To compile custom tensorflow ops 