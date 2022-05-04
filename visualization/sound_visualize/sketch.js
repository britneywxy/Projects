
let mic;

let flowerPetalAngle = 0;
let angleStep = 15;

const MIN_PETAL_LENGTH = 90;
const MAX_PETAL_LENGTH = 450;

let bgColorSolid = null;
let bgColorTranslucent = null;

//let currentXPos = 0;
//let currentYPos = 0;

//var angle = 0;	// initialize angle variable
//var scalar = 50;  // set the radius of circle
//var startX = 250;	// set the x-coordinate for the circle center
//var startY = 200;	// set the y-coordinate for the circle center

function setup() {
  createCanvas(600, 400);
  
  // Gets a reference to computer's microphone
  // https://p5js.org/reference/#/p5.AudioIn
  mic = new p5.AudioIn();

  // Start processing audio input
  // https://p5js.org/reference/#/p5.AudioIn/start
  mic.start();
  
  // Helpful for debugging
  printAudioSourceInformation();
  angleMode(DEGREES); 

  // frameRate(20);
  bgColorSolid = color(10);
  bgColorTranslucent = color(30, 1);
  background(bgColorSolid);
}

function draw() {
  //background(5, 5, 5, 1);
  background(bgColorTranslucent);

  // get current microphone level (between 0 and 1)
  // See: https://p5js.org/reference/#/p5.AudioIn/getLevel
  let micLevel = mic.getLevel(); // between 0 and 1
  push();
  colorMode(HSB);
  rectMode(CENTER);
  
  ellipseMode(CENTER);
  
  
  //var x = startX + scalar * cos(angle);
  //var y = startY + scalar * sin(angle);
  
  //translate(x, y,8);
  translate(width/2,height/2);
  
  
  //angle++;
 
  const hue = flowerPetalAngle;
  stroke(hue, 100, 100, 0.6);
  fill(hue, 50, 80, 0.2);
  rotate(flowerPetalAngle);
  const petalLength = map(micLevel, 0, 1, MIN_PETAL_LENGTH, MAX_PETAL_LENGTH);
  ellipse(50, 10, petalLength, 20);
  //angle++;
  
  pop();

  flowerPetalAngle += angleStep;

  if(flowerPetalAngle > 360){
    flowerPetalAngle = 0;
    //flowerPetalAngle = random(0, 7);
    //angleStep = random(10, 35);
  }
  drawFps();
}


function drawFps(){
  // Draw fps
  push();
  const fpsLblTextSize = 8;
  textSize(fpsLblTextSize);
  const fpsLbl = nf(frameRate(), 0, 1) + " fps";
  const fpsLblWidth = textWidth(fpsLbl);
  const xFpsLbl = 4;
  const yFpsLbl = 10;
  fill(30);
  noStroke();
  rect(xFpsLbl - 1, yFpsLbl - fpsLblTextSize, fpsLblWidth + 2, fpsLblTextSize + textDescent());

  fill(150);
  text(fpsLbl, xFpsLbl, yFpsLbl);
  pop();
}

function printAudioSourceInformation(){
  let micSamplingRate = sampleRate();
  print(mic);

  // For debugging, it's useful to print out this information
  // https://p5js.org/reference/#/p5.AudioIn/getSources
  mic.getSources(function(devices) {
    print("Your audio devices: ")
    devices.forEach(function(device) {
      print("  " + device.kind + ": " + device.label + " id = " + device.deviceId);
    });
  });
  print("Sampling rate:", sampleRate());

  // Helpful to determine if the microphone state changes
  getAudioContext().onstatechange = function() {
    print("getAudioContext().onstatechange", getAudioContext().state);
  }
}