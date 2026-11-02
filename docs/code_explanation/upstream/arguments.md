# `arguments/__init__.py`

## Purpose

The configuration system: parameter groups, command-line parsing, and the merge of
command-line arguments with a saved `cfg_args` file.

**This file is the direct cause of both Stage 1 failures.** Understanding it is what
turned two confusing tracebacks into a single correct fix.

## Line-by-line

### Lines 16–17 — `GroupParams`

```python
class GroupParams:
    pass
```

An empty container. Attributes are attached dynamically by `extract`, which is why
a missing field produces `AttributeError: 'GroupParams' object has no attribute
'depths'` rather than a clear configuration error — the object simply never received
that attribute.

### Lines 19–37 — `ParamGroup.__init__`

```python
class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)
```

Turns instance attributes into command-line arguments by reflection. An attribute
named `_source_path` becomes `--source_path` **plus** the short form `-s` (line 30),
which is why `render.py` is invoked with `-m` and `-s`.

**Line 28 is the mechanism behind the second bug:**

```python
            value = value if not fill_none else None
```

When `fill_none` is true, **every default becomes `None`** instead of the declared
value.

### Lines 39–44 — `extract`

```python
    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group
```

Copies matching attributes from the merged namespace into a `GroupParams`. Note it
only sets attributes that **exist in the merged namespace** — if a field was never
supplied, it is simply absent, and the error surfaces later at the point of use
(in `scene/__init__.py` line 44, for `depths`).

### Lines 45–59 — `ModelParams`

```python
class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)
```

**The complete field list our `fix_cfg_args.py` must reproduce.** Ten fields:
`sh_degree`, `source_path`, `model_path`, `images`, `depths`, `resolution`,
`white_background`, `train_test_exp`, `data_device`, `eval`.

`_depths` and `train_test_exp` are the two added after the 2023 checkpoint was
produced — the missing fields that caused the `AttributeError`.

Line 59 passes `sentinel` through as `fill_none`. `render.py` calls
`ModelParams(parser, sentinel=True)`, so **all ten defaults become `None`**.

### Lines 66–72 — `PipelineParams`

```python
class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")
```

Rendering options. Note **no** `sentinel` — these keep their real defaults.

`convert_SHs_python = False` means SH evaluation happens in CUDA, not Python — the
branch discussed in [gaussian_renderer.md](gaussian_renderer.md). This is what our
renders use, identically for every configuration.

### Lines 74–100 — `OptimizationParams` (unused)

Training hyperparameters. Not used by this project, but two are informative context
for the compression findings:

- `feature_lr = 0.0025` — the SH learning rate; `gaussian_model.py` applies **1/20th**
  of this to higher-order coefficients.
- `iterations = 30_000` — matching our checkpoint's iteration count.

### Lines 102–122 — `get_combined_args`: **the source of both bugs**

```python
def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)
```

Lines 107–112 read the model directory's `cfg_args` as **text**. The two `print`
statements are the familiar "Looking for config file in… / Config file found:" lines
in every Colab log.

**Line 116 is the first bug.** `eval(cfgfile_string)` executes the file's contents as
a Python expression. The checkpoint's file contained:

```python
Namespace(sh_degree=3, source_path='C:\Users\PC\Desktop\cloud\train', ...)
```

Inside a string literal, `\U` begins an eight-hex-digit unicode escape. `\Users` is
malformed, so the parse fails before any rendering:

```
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes
in position 2-3: truncated \UXXXXXXXX escape
```

Note also that `except TypeError` catches only the case where `model_path` is `None`
— a *missing* file raises `FileNotFoundError` and is not caught, and a *malformed*
file raises `SyntaxError` from `eval`, also uncaught.

```python
    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
```

**Lines 118–122 are the second bug.**

The merge starts from the config file's values (line 118), then overrides with
command-line values — but **only where the command-line value is not `None`**
(line 120).

Because `render.py` uses `sentinel=True`, every model default *is* `None`. So no
`ModelParams` field is ever contributed by the command line, and the merged namespace
contains exactly the keys present in `cfg_args`.

A `cfg_args` missing `depths` therefore produces a namespace with no `depths` key,
`extract` never sets the attribute, and the failure appears much later as
`AttributeError` inside `scene/__init__.py`.

The design intent is reasonable — a saved model's own configuration should win over
stale command-line defaults — but it makes `cfg_args` completeness a hard requirement,
with a failure mode that points at the wrong file.

## Why it matters

This file explains why `scripts/fix_cfg_args.py` takes the form it does:

- It writes **POSIX paths**, because line 116 will `eval()` them and backslashes
  become escape sequences.
- It writes the **complete ten-field set**, because lines 118–122 mean nothing is
  supplied from anywhere else.
- It sets **`eval=True`**, because that value flows to `scene/__init__.py` line 44 and
  determines whether a test split exists at all.

For the report, this is a good worked example of diagnosis over workaround: the
tracebacks pointed at `render.py` and `scene/__init__.py`, but the cause was here, in
how configuration is merged. One fix addressed all three problems.
