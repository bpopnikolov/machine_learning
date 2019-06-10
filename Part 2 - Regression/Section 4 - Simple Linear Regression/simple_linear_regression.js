const ms = require('modelscript/build/modelscript.cjs.js');
const np = require('numjs');
const ml = require('ml');
const pd = require('pandas-js');
const plt = require('matplotnode');
const { PythonShell } = require('python-shell');
const path = require('path');

const callPythonPlot = (...args) => {
  const options = {
    mode: 'text',
    pythonPath: '../../../../../../../usr/bin/python3',
    pythonOptions: ['-u'], // get print results in real-time
    scriptPath: path.resolve(__dirname),
    args: [...args]
  };

  const pyshell = PythonShell.run(
    './node_plotter.py',
    options,
    (err, results) => {
      if (err) throw err;
      console.log('finished:');
      console.log(results);
    }
  );
};

(async () => {
  const csvData = await ms.loadCSV('./Salary_Data.csv');
  const dataset = new ms.DataSet(csvData);

  // const X = dataset.columnMatrix(['YearsExperience']);
  const X = dataset.columnArray('YearsExperience');
  const y = dataset.columnArray('Salary');

  const { train: X_train, test: X_test } = ms.cross_validation.train_test_split(
    X,
    {
      test_size: 1 / 3,
      random_state: 0
    }
  );

  const { train: y_train, test: y_test } = ms.cross_validation.train_test_split(
    y,
    {
      test_size: 1 / 3,
      random_state: 0
    }
  );

  // pandas DataFrames
  // const dataset = new pd.DataFrame(fittedData);
  // const X = dataset.iloc([0, dataset.length], [0, 5]).values;
  // const y = dataset.iloc([0, dataset.length], [5, 6]).values;

  //   console.log({
  //     X_train,
  //     X_test,
  //     y_train,
  //     y_test
  //   });

  const progressor = new ml.SimpleLinearRegression(X_train, y_train);
  const y_predict = progressor.predict(X_test);
  const y_predict_train = progressor.predict(X_train);

  callPythonPlot(X_train, y_train, y_predict_train);
})();
