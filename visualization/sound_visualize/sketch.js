/*
 Sound Visualization

  In this visualization, I use the main huge star to visualize the waveform by changing its size, and the particles are related to the frequency. When the frequency reach a specific level, the particles will move differently. 
  
  I use the amplitude level of the sound to change the rotation degree. So the rotation of the star seems more consistent with the song.
  
  I also use the color because I think color will make the visualization have a sense of rhythm with the song

*/

var song 
var fft 
let starAngle = 0 // rotating angle
var particles = [] //store the particles array 
var color // the star and the particles will have the consistent color
var amplitude // the amplitude 

function preload(){
  //song = loadSound('normal.mp3') //load the song
  song = loadSound('everglow.mp3')
}

function setup() {
  createCanvas(windowWidth, windowHeight);
  fft = new p5.FFT()
  amplitude = new p5.Amplitude();
  angleMode(DEGREES)
}

function draw() {
  background(0);
  this.color = [random(0,255), random(0,255), random(0,255)]
  color = this.color
  stroke(this.color) // change the color to see the wave
  strokeWeight(2)
  fill(0)
  rectMode(CENTER);
  
  
  // store the waveform data
  var wave = fft.waveform()
  
  // amplitude level
  let level = amplitude.getLevel()
  
  push()
  translate(width/2, height/2)
  fft.analyze()
  amp = fft.getEnergy(20,200) //frequency
  rotate(starAngle)
  //star(0, 0, 30, 70, 5, wave);
  let angle = 360 / 5;
  let halfAngle = angle / 2.0;
  beginShape();
  for (let a = 0; a <= 360; a += angle) {
    // map the variable to the index of wave, the value should be an int
    var index = floor(map(a, 0, 360, 0, wave.length - 1)) 
    // radius from 0 degree to 360 degree
    var r = map(wave[index], -2, 2, 0, 360) 
    var sx = cos(a) * r;
    var sy = sin(a) * r;
    vertex(sx, sy); // points of the star
    sx = cos(a + halfAngle) * 70;
    sy = sin(a + halfAngle) * 70;
    vertex(sx, sy); // points of the star
    
    // let the rotation degree changes with the amplitude level
    starAngle += 3*level 
  }
  endShape(CLOSE);
  
  //create a new particle every frame
  var p = new Particle()
  particles.push(p)
  
  for (var i = 0; i < particles.length; i++) {
    particles[i].update(amp > 200) 
    particles[i].show() // show every particle
  }
  
  pop()
  
  
}


function mouseClicked() {
  if(song.isPlaying()){
    song.pause()
    noLoop() // freeze when pause
  } else {
    song.play()
    loop()
  }
}

// make the particles 
class Particle {
  constructor() {
    this.pos = p5.Vector.random2D().mult(120) // place random particles around the waveform    
    // let the particles move --> they should have velocity and accerleration
    this.vel = createVector(0,0)
    this.acc = this.pos.copy().mult(random(0.0001, 0.00001)) //accerleration should have the same direction as the position vector       
    this.w = random(3,5)
    this.color = [random(0,255), random(0,255), random(0,255),]
  }
  update(cond) { // update the variables
    this.vel.add(this.acc)
    this.pos.add(this.vel)
    if (cond) {
      // add the velocity to the position a couple more times
      // this will make the particles respond to the music
      this.pos.add(this.vel)
      this.pos.add(this.vel)
      this.pos.add(this.vel) 
      this.pos.add(this.vel) 
    }
  }
  show(){
    noStroke()
    fill(color);
    ellipse(this.pos.x, this.pos.y, this.w)
  }
}
