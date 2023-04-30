#lang racket
(require math/matrix)

; Activation functions and their derivatives
(define (sigmoid x) (/ (exp x) (+ (exp x) 1)))
(define (sigmoid~ x) (* (sigmoid x) (- 1 (sigmoid x))))

(define (tanh x) (tanh x))
(define (tanh~ x) (/ 1 (expt (cosh x) 2)))

(define (relu x) (max x 0))
(define (relu~ x) (if (> x 0) 1 0))


(define activation sigmoid)
(define activation~ sigmoid~)

#|
(define (sigmoid x)
  (/ 1 (+ 1 (exp (- x)))))

(define (sigmoid-derivative x)
  (* (sigmoid x) (- 1 (sigmoid x))))


(define (dot-product a b)
  (apply + (map * a b)))

(define (matrix-vector-multiply m v)
  (map (lambda (row) (dot-product row v)) m))

(define (matrix-multiply a b)
  (apply map list (map (lambda (row) (matrix-vector-multiply b row)) a)))

(define (transpose m)
  (apply map list m))

(define (add-bias inputs)
  (cons 1 inputs))

(define (remove-bias inputs)
  (cdr inputs))

|#

; Forward pass
(define (feedforward input weights biases)
  (if (empty? weights)
      (list)
      (let* ([z (matrix+ (matrix* (first weights) input) (first biases))]
             [out (matrix-map activation z)])
        (cons (list z out)
              (feedforward out (rest weights) (rest biases))))))

; Backpropagation

(define (backprop weights outputs expected-output)
  (let loop ([errors (list)]
             [prev-error (output-error (first (last outputs)) (second (last outputs)) expected-output)]
             [outputs (reverse (drop-right (reverse outputs) 1))]
             [weights (reverse (rest weights))])
    (if (empty? outputs)
        (reverse (cons prev-error errors))
        (let ([error (matrix-map * (matrix* (first weights) prev-error) (matrix-map activation~ (first (first outputs))))])
          (loop (cons prev-error errors) error (rest outputs) (rest weights))))))

; Gradient descent
(define (gradient-descent weights biases inputs expected-output (n 1))
  (let* ([outputs (feedforward inputs weights biases)]
         [errors (backprop weights outputs expected-output)]
         [parameters (adjust-parameters weights biases outputs errors)]
         [weights (map (lambda (x) (first x)) parameters)]
         [biases (map (lambda (x) (second x)) parameters)])
    (if (<= n 1)
        (values weights biases)
        (gradient-descent weights biases inputs expected-output (sub1 n)))))




; Helper functions
(define (output-error z actual expected)
  (matrix-map * (matrix- actual expected) (matrix-map activation~ z)))


(define (adjust-weights weight out err epsilon)
  (let-values ([(w h) (matrix-shape weight)])
    (let loop ([j 0] [rows '()])
      (if (= j w)
          (list->matrix w h (append* (reverse rows)))
          (loop (+ j 1)
                (cons (let loop2 ([k 0] [row '()])
                        (if (= k h)
                            (reverse row)
                            (loop2 (+ k 1)
                                   (cons (- (matrix-ref weight j k)
                                            (* (matrix-ref out k 0) (matrix-ref err j 0) epsilon))
                                         row))))
                      rows))))))

(define (adjust-parameters weights biases outputs errors)
  (let ([epsilon 1])
    (map (lambda (w b o e) (list (adjust-weights w (second o) e epsilon) (matrix- b (matrix-scale e epsilon))))
         weights biases outputs errors)))

; Example usage
(define weights (list (matrix ((0.5 0.2) (0.3 0.6)))
                      (matrix ((0.1 0.8) (0.7 0.4)))))
(define biases (list (col-matrix (0.1 0.2))
                     (col-matrix (0.3 0.4))))

(define inputs (col-matrix (0.9 0.8)))
(define expected-output (col-matrix (0.5 0.25)))

(define (set-activation-function func)
  (cond
    [(eq? func 'sigmoid)
     (begin
       (set! activation sigmoid)
       (set! activation~ sigmoid~))]
    [(eq? func 'tanh)
     (begin
       (set! activation tanh~)
       (set! activation~ tanh~))]
    [(eq? func 'relu)
     (begin
       (set! activation relu)
       (set! activation~ relu~))]
     (else (error "Unsupported activation function"))))




#|

(set-activation-function 'sigmoid)

(let-values ([(trained-weights trained-biases) (gradient-descent weights biases inputs expected-output 256)])
  (display (second (last (feedforward inputs trained-weights trained-biases)))) (newline))


The output you're seeing, #<array #(2 1) #[0.17129497823351528 0.6856399087738091]>,
represents the final output of the trained MLP given the input. This output is a 2x1
matrix (or column vector) with the values 0.17129497823351528 and 0.6856399087738091.

This output is the result of the feedforward process after training the MLP using
gradient descent for 256 iterations. It shows the network's prediction based on the
input provided ((col-matrix (0.9 0.8))). In the given code, the network was trained
to minimize the difference between its output and the expected output ((col-matrix (0.5 0.25))).

|#
