# `mcrotk_utils.pandas_utils`

`pandas` helpers: making, concatenating, merging, and adding to DataFrames, and dtypes for their columns. It needs `pandas` (`mcrotk-utils[pandas]`). Each function's docstring has the details: its arguments, result, and errors.

[Back to the package README](../../../README.md)

## Contents

The names are grouped below by what they're for.

### Making DataFrames

These build DataFrames from other sources.

- `as_dataframe`: a DataFrame as it is, or a CSV file (a path or an open file) read into one, for functions that accept either.
- `matrices_to_long_df`: named matrices of the same shape into one long DataFrame, with a row per matrix position and a column per matrix.

### Dtypes

These set the types of columns.

- `column_dtypes`: a mapping of columns to categorical and nullable integer dtypes, to pass to `df.astype`.
- `to_datetime_columns`: convert columns of one or more DataFrames to datetimes, in place.

### Combining DataFrames

These do what `pd.concat` and `pd.merge` do, while keeping dtypes and combining columns.

- `combine_dataframes`: concatenate DataFrames, keeping a categorical column categorical when its categories differ between them (`pd.concat` makes it a plain column).
- `merge_with_preferences`: merge two DataFrames on their indexes, combining chosen columns that both have into one, with the values from a preferred side.

### Adding columns

These add columns computed from others.

- `join_columns`: combine several columns into one column of labels such as `"(a, 1)"`, e.g. to group or plot by them.
- `add_lags`: add lagged copies of the columns, holding each row's values from 1, 2, ... periods earlier.

### Binary columns

These check whether a Series holds 0/1 values (with `True` and `False` counting as 1 and 0).

- `is_binary`: whether every value is 0 or 1.
- `num_non_binary`: how many values aren't 0 or 1.

## Examples

These examples show the two functions for combining DataFrames.

### `combine_dataframes`

Results from separate runs often have categorical columns with different categories. Here, each run saw different cities:

```pycon
>>> import pandas as pd
>>> from mcrotk_utils.pandas_utils import combine_dataframes, merge_with_preferences
>>> first = pd.DataFrame({"city": pd.Categorical(["Oslo", "Rome"]), "sales": [3, 5]})
>>> first
   city  sales
0  Oslo      3
1  Rome      5
>>> second = pd.DataFrame({"city": pd.Categorical(["Lima"]), "sales": [4]})
>>> second
   city  sales
0  Lima      4
```

`pd.concat` would turn the city column into plain strings. `combine_dataframes` keeps it categorical, with all the categories, the first DataFrame's first:

```pycon
>>> combined = combine_dataframes(first, second, ignore_index=True)
>>> combined
   city  sales
0  Oslo      3
1  Rome      5
2  Lima      4
>>> combined["city"].dtype.name, list(combined["city"].cat.categories)
('category', ['Oslo', 'Rome', 'Lima'])
```

### `merge_with_preferences`

Here a product catalog and a price feed overlap in some products, and both have names and prices. The catalog's names are the ones to keep, but the feed's prices are newer:

```pycon
>>> catalog = pd.DataFrame(
...     {"name": ["Apple", "Banana", "Cherry"], "aisle": ["A1", "A1", "B2"], "price": [1.00, 0.50, 3.00]},
...     index=[1, 2, 3],
... )
>>> catalog
     name aisle  price
1   Apple    A1    1.0
2  Banana    A1    0.5
3  Cherry    B2    3.0
>>> feed = pd.DataFrame(
...     {"name": ["banana", "cherry", "date"], "price": [0.60, 3.50, 2.00], "supplier": ["Ross", "Ross", "Vale"]},
...     index=[2, 3, 4],
... )
>>> feed
     name  price supplier
2  banana    0.6     Ross
3  cherry    3.5     Ross
4    date    2.0     Vale
```

A plain merge keeps both versions of each shared column, leaving them to be combined by hand, column by column:

```pycon
>>> catalog.merge(feed, how="outer", left_index=True, right_index=True)
   name_x aisle  price_x  name_y  price_y supplier
1   Apple    A1      1.0     NaN      NaN      NaN
2  Banana    A1      0.5  banana      0.6     Ross
3  Cherry    B2      3.0  cherry      3.5     Ross
4     NaN   NaN      NaN    date      2.0     Vale
```

`merge_with_preferences` combines them in one call: each product in both takes the catalog's name and the feed's price, and each product in only one takes what that one has. The combined columns come after the others:

```pycon
>>> merge_with_preferences(catalog, feed, "outer", preference_left=["name"], preference_right=["price"])
  aisle supplier    name  price
1    A1      NaN   Apple    1.0
2    A1     Ross  Banana    0.6
3    B2     Ross  Cherry    3.5
4   NaN     Vale    date    2.0
```

It also keeps the combined columns' dtypes where both DataFrames share them, including categoricals (with both sides' categories), and can rename columns and convert date columns first; see its docstring.
