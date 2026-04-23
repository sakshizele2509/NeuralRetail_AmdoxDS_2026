def validate_data(df):

    assert df["CustomerID"].isnull().sum() == 0
    assert (df["Quantity"] > 0).all()
    assert (df["UnitPrice"] > 0).all()

    print("Data validation passed.")