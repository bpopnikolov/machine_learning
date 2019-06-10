const ms = require('modelscript/build/modelscript.cjs.js');
const np = require('numjs');
const ml = require('ml');
const pd = require('pandas-js');
const mpn = require('matplotnode');

(async () => {
  const csvData = await ms.loadCSV('./Data.csv');
  const dataset = new ms.DataSet(csvData);
  dataset.fitColumns({
    columns: [
      { name: 'Age', strategy: 'mean' },
      { name: 'Salary', strategy: 'mean' },
      { name: 'Country', options: { strategy: 'label' } },
      { name: 'Country', options: { strategy: 'onehot' } },
      {
        name: 'Purchased',
        options: { strategy: 'label', labelOptions: { binary: true } }
      }
    ]
  });

  const X = dataset.columnMatrix([
    ['Country_France'],
    ['Country_Germany'],
    ['Country_Spain'],
    ['Age', { scale: 'standard' }],
    ['Salary', { scale: 'standard' }]
  ]);
  console.log(X);
  const y = dataset.columnArray('Purchased');

  const { train: X_train, test: X_test } = ms.cross_validation.train_test_split(
    X,
    {
      test_size: 0.2,
      random_state: 0
    }
  );

  const { train: y_train, test: y_test } = ms.cross_validation.train_test_split(
    y,
    {
      test_size: 0.2,
      random_state: 0
    }
  );

  // pandas DataFrames
  // const dataset = new pd.DataFrame(fittedData);
  // const X = dataset.iloc([0, dataset.length], [0, 5]).values;
  // const y = dataset.iloc([0, dataset.length], [5, 6]).values;

  console.log({
    X_train,
    X_test,
    y_train,
    y_test
  });
})();
