package com.weka;

import java.io.File;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.*;
import weka.classifiers.trees.M5P;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class RegressionTask {
	public static void main(String args[]) {
		try {
		/*
		 * Load data
		 */
		CSVLoader loader = new CSVLoader();
		loader.setFieldSeparator(",");
			File file = new File("src/main/resources/data/ENB2012_data.csv");
			loader.setSource(file);
		Instances data = loader.getDataSet();

		// System.out.println(data);

		/*
		 * Build regression models
		 */
		// set class index to Y1 (heating load)
		data.setClassIndex(data.numAttributes() - 2);
		// remove last attribute Y2
		Remove remove = new Remove();
		remove.setOptions(new String[] { "-R", data.numAttributes() + "" });
		remove.setInputFormat(data);
		data = Filter.useFilter(data, remove);

		// build a regression model
			ClassLoader classLoader = RegressionTask.class.getClassLoader();
			Class aClass = null;
			try {
				 aClass = classLoader.loadClass("weka.classifiers.functions.LinearRegression");
				System.out.println("aClass.getName() = " + aClass.getName());
				
			} catch (ClassNotFoundException e)
			{
				e.printStackTrace();
			}
		LinearRegression model=(LinearRegression) aClass.newInstance();
		//LinearRegression model = new LinearRegression();
		model.buildClassifier(data);
		System.out.println(model);

		// 10-fold cross-validation
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(model, data, 10, new Random(1), new String[] {});
		System.out.println(eval.toSummaryString());
		double coef[] = model.coefficients();
		System.out.println();

		// build a regression tree model

		M5P md5 = new M5P();
		md5.setOptions(new String[] { "" });
		md5.buildClassifier(data);
		System.out.println(md5);

		// 10-fold cross-validation
		eval.crossValidateModel(md5, data, 10, new Random(1), new String[] {});
		System.out.println(eval.toSummaryString());
		System.out.println();
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
}
