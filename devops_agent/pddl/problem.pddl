(define (problem devops-p)
  (:domain devops)

  (:objects
    svc_e - service
    bad good - configopt)

  (:init
    (= (mem_cost svc_e) 512)
    (= (total-cost) 0)
    (config_ok svc_e good)
  )

  (:goal (running svc_e))

  (:metric minimize (total-cost)))
