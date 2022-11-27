import * as ort from 'onnxruntime-web';
import _ from 'lodash';
import { imagenetClasses } from '../data/imagenet';

export async function runSqueezenetModel(preprocessedData: any): Promise<[any, number]> {
  
  // // Create session and set options. See the docs here for more options: 
  // //https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#graphOptimizationLevel
  // const session = await ort.InferenceSession
  //                         .create('./_next/static/chunks/pages/squeezenet1_1.onnx', 
  //                         { executionProviders: ['webgl'], graphOptimizationLevel: 'all' });
  
  // // [NOT WORKING] YOLOv7-tiny 416x416
  // const session = await ort.InferenceSession
  //                         .create('./_next/static/chunks/pages/yolov7-tiny-416-best.onnx');

  // [WORK!!] YOLOv7-tiny 640x640
  const session = await ort.InferenceSession
                          .create('./_next/static/chunks/pages/yolov7-tiny-640-29epochs.onnx');


  console.log('Inference session created')
  // Run inference and get results.
  var [results, inferenceTime] =  await runInference(session, preprocessedData);
  return [results, inferenceTime];
}

async function runInference(session: ort.InferenceSession, preprocessedData: any): Promise<[any, number]> {
  // Get start time to calculate inference time.
  const start = new Date();
  // create feeds with the input name from model export and the preprocessed data.
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = preprocessedData;
  
  // // Run the session inference.
  // const outputData = await session.run(feeds);
  // // Get the end time to calculate inference time.
  // const end = new Date();
  // // Convert to seconds.
  // const inferenceTime = (end.getTime() - start.getTime())/1000;
  // // Get output results with the output name from the model export.
  // const output = outputData[session.outputNames[0]];
  // //Get the softmax of the output data. The softmax transforms values to be between 0 and 1
  // var outputSoftmax = softmax(Array.prototype.slice.call(output.data));
  
  // //Get the top 5 results.
  // var results = imagenetClassesTopK(outputSoftmax, 5);
  // console.log('results: ', results);

  const { output } = await session.run(feeds);
  const end = new Date();
  const inferenceTime = (end.getTime() - start.getTime())/1000;
  const boxes = [];
  const labels = [
    "onepack",
    "fruit_apple",
    "bottle",
    "person",
    "cell_phone"
  ]  

  const results: any = []

  // looping through output
  for (let r = 0; r < output.size; r += output.dims[1]) {
    const data = output.data.slice(r, r + output.dims[1]); // get rows
    const [x0, y0, x1, y1, classId, score] = data.slice(1);
    const w = x1 - x0,
      h = y1 - y0;

    console.log(`${labels[classId]} : ${score}`)

    boxes.push({
      classId: classId,
      probability: score,
      bounding: [x0, y0, w, h],
    });

    if(boxes.length > 0) {
      boxes.map(box => {
        results.push({ 
          id:-1, 
          index:-1, 
          name:labels[box.classId], 
          probability: score
        })
      })
    }
  }

  console.log('results:', results)
  
  return [results, inferenceTime];
}

//The softmax transforms values to be between 0 and 1
function softmax(resultArray: number[]): any {
  // Get the largest value in the array.
  const largestNumber = Math.max(...resultArray);
  // Apply exponential function to each result item subtracted by the largest number, use reduce to get the previous result number and the current number to sum all the exponentials results.
  const sumOfExp = resultArray.map((resultItem) => Math.exp(resultItem - largestNumber)).reduce((prevNumber, currentNumber) => prevNumber + currentNumber);
  //Normalizes the resultArray by dividing by the sum of all exponentials; this normalization ensures that the sum of the components of the output vector is 1.
  return resultArray.map((resultValue, index) => {
    return Math.exp(resultValue - largestNumber) / sumOfExp;
  });
}
/**
 * Find top k imagenet classes
 */
export function imagenetClassesTopK(classProbabilities: any, k = 5) {
  const probs =
      _.isTypedArray(classProbabilities) ? Array.prototype.slice.call(classProbabilities) : classProbabilities;

  const sorted = _.reverse(_.sortBy(probs.map((prob: any, index: number) => [prob, index]), (probIndex: Array<number> ) => probIndex[0]));

  const topK = _.take(sorted, k).map((probIndex: Array<number>) => {
    const iClass = imagenetClasses[probIndex[1]];
    return {
      id: iClass[0],
      index: parseInt(probIndex[1].toString(), 10),
      name: iClass[1].replace(/_/g, ' '),
      probability: probIndex[0]
    };
  });
  return topK;
}

