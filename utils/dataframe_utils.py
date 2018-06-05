def statistic_df(self):
    """
    This returns a describe() and a null analysis of the file used as input.
    Also, if output = True, it returns two CSV files in doc_output\statistics
    wit names describe.csv and null.csv
    """

    print('DESCRIBE-------------------------')
    describe_df = self.describe(include='all')
    print(describe_df)
    print(' ')

    print('NULL VALUES-----------------------')

    null = self.isnull().sum()
    print(null)

    print(' ')
    print('INFO-----------------------------')
    print(self.info())
    print(' ')