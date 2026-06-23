(define (problem devops-p)
  (:domain devops)

  (:objects
    svc_e - service
    bad good - configopt)

  (:init
    (config_ok svc_e good)
  )

  (:goal (running svc_e))
)
