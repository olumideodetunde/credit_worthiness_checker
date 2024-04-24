import pandas as pd

class Transform:
    def __init__(self, df:pd.DataFrame):
        self.df = df
    
    def engineer_family_size(self, col: str, new_col_name: str, bins: list, labels: str) -> pd.DataFrame:
        """This method calculates the family size based on the number of children"""
        self.df[str(col) + "_binned"] = pd.cut(self.df[col], bins=bins, labels=labels,
                                                include_lowest=True)
        self.df.rename(columns={col + "_binned": new_col_name}, inplace=True)
        return self.df
    
    def standardise_feature(
        self, cols: list, mldatatype: str, path: str) -> pd.DataFrame:
        """This method standardises the numerical features in the dataframe"""
        scaler = StandardScaler()
        if mldatatype == "test":
            for col in cols:
                col_data = self.df[[col]]
                scaler.fit(col_data)
                self.df[str(col) + "_standardised"] = scaler.transform(col_data)
            return self.df
        else:
            raise ValueError("Invalid mldatatype")
                
    def one_hot_encode(self, cat_col_list: list, path:str) -> pd.DataFrame:
        """This method transforms the categorical columns to be ml ready"""
        encoder = OneHotEncoder(sparse_output=False)
        cat_encoded = encoder.fit_transform(self.df[cat_col_list])
        cat_columns = []
        for i, col in enumerate(cat_col_list):
            for category in encoder.categories_[i]:
                cat_columns.append(f"{col}_{category}")
        cat_encoded = pd.DataFrame(cat_encoded, columns=cat_columns)
        self.df = pd.concat([self.df, cat_encoded], axis=1)
        return self.df
    
    def create_inference_df(self, required_features:list) -> pd.DataFrame:
        """This method creates a dataframe with the required features for inference"""
        for col in required_features:
            if col not in self.df.columns:
                self.df[col] = 0
        return self.df[required_features]