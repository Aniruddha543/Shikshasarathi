let slideIndex = 0;
showSlides();

function showSlides() {
  let slides = document.getElementsByClassName("slide");
  let dots = document.getElementsByClassName("dot");

  // Hide all slides
  for (let i = 0; i < slides.length; i++) {
    slides[i].style.display = "none";
  }

  // Reset dots
  for (let i = 0; i < dots.length; i++) {
    dots[i].className = dots[i].className.replace(" active-dot", "");
  }

  // Move to next slide
  slideIndex++;
  if (slideIndex > slides.length) { slideIndex = 1 }

  slides[slideIndex-1].style.display = "block";
  dots[slideIndex-1].className += " active-dot";

  // Auto change every 5s
  setTimeout(showSlides, 5000);
}

// Manual dot navigation
function currentSlide(n) {
  slideIndex = n-1;
  showSlides();
}
