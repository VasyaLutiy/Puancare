(define (problem devops-p0)
  (:domain devops)

  (:objects
    svc_a - service
    cfg0  - configopt
  )

  (:init
    (= (min_safe_mem svc_a) 512)
    (= (total-cost) 0)
  )

  (:goal
    (running svc_a)
  )

  (:metric minimize (total-cost))
)
