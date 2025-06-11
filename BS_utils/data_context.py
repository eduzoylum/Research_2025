import polars as pl

class DataContext:
    """
    Lightweight wrapper for a Polars DataFrame with flexible column aliasing.
    
    Parameters
    ----------
    df : pl.DataFrame
        Source data (options, quotes, etc.)
    col_map : dict
        Maps logical column names (e.g., 'bid', 'ask') to actual column names
    """
    def __init__(self, df: pl.DataFrame, col_map: dict):
        self.df = df
        self.col_map = col_map

    def get(self, key: str) -> pl.Series:
        """Returns the Series corresponding to the logical column name `key`."""
        actual = self.col_map[key]
        return self.df[actual]

    def set(self, key: str, value) -> None:
        """
        Set or replace a column under a logical name. Automatically adds to col_map if new.
        Parameters:
            key (str): Logical column name.
            value: A Polars Series or compatible object to be added as a new column.
        """
        col_name = self.col_map.get(key, key)
        self.df = self.df.with_columns(pl.Series(name=col_name, values=value))
        self.col_map[key] = col_name

    def rename_column(self, key: str, new_col_name: str) -> None:
        """Updates the mapping for `key` to use a new column name."""
        self.col_map[key] = new_col_name

    def has(self, key: str) -> bool:
        """Checks whether a logical key is mapped."""
        return key in self.col_map

    def require(self, *keys: str) -> None:
        """Ensures that all required logical keys exist."""
        missing = [k for k in keys if k not in self.col_map]
        if missing:
            raise KeyError(f"Missing required columns in col_map: {missing}")

    def raw(self) -> pl.DataFrame:
        """Returns the raw underlying Polars DataFrame."""
        return self.df
    
    def update_column(self, key, func, return_dtype=None):
        """
        Applies a transformation function to a column in-place using .map_elements
        and updates the DataFrame.
        Parameters:
            key (str): Logical column name.
            func (callable): Function to apply to each element.
            return_dtype (pl.DataType, optional): Polars return dtype. If not provided,
                                                the existing column dtype is used.
        """
        if key not in self.col_map:
            raise KeyError(f"'{key}' not in column mapping.")
        col = self.col_map[key]
        current_dtype = self.df.schema[col]
        if return_dtype is None:
            return_dtype = current_dtype
        self.df = self.df.with_columns(
            self.df[col].map_elements(func, return_dtype=return_dtype).alias(col)
        )

    def to_datetime(self, key: str, **kwargs) -> None:
        """
        Converts the specified column to datetime.
        Parameters:
            key (str): Logical column name.
            **kwargs: Passed to `pl.col(...).str.strptime(...)` if needed.
        """
        col = self.col_map[key]  
        if self.df[col].dtype != pl.Datetime:
            self.df = self.df.with_columns(
                self.df[col].str.strptime(pl.Datetime, **kwargs).alias(col)
            )

    def filter(self, condition: pl.Expr) -> 'DataContext':
        """
        Returns a new DataContext filtered by a given logical column name and condition.
        Parameters:
            condition (pl.Expr): Polars expression, e.g., pl.col('T') > 0.
        Returns:
            DataContext: A new instance with filtered rows.
        """       
        return DataContext(self.df.filter(condition), self.col_map.copy())

    def select(self, *keys: str) -> pl.DataFrame:
        """
        Returns a new Polars DataFrame selecting columns by logical names.
        """
        actual_cols = [self.col_map[k] for k in keys]
        return self.df.select(actual_cols)

    def col(self, key: str) -> pl.Expr:
        """
        Returns a Polars expression for use in filters, selects, etc.
        """
        return pl.col(self.col_map[key])

    def group_by(self, keys: list[str]):
        """
        Returns a Polars GroupBy object using logical column names.
        """
        actual_cols = [self.col_map[k] for k in keys]
        return self.df.group_by(actual_cols)

    def describe(self) -> None:
        print(self.df.describe())

    def show(self, n: int = 10) -> None:
        print(self.df.head(n))

    def shape(self):
        return self.df.shape
    
    def is_empty(self):
        """Check if the DataFrame is empty."""
        return self.df.is_empty()
