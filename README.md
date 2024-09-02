1. Install dependecies:

```
./deps.sh
```

2. Clean and normalize raw data:

```
./prepare.py --verbose <data-file.txt>
```

3. Training

```
./train.py normalized.csv
```
