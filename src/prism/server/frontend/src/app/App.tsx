import { Navigate, Route, Routes } from 'react-router-dom'

import { AppShell } from '../components/AppShell'
import DashboardPage from '../pages/DashboardPage'
import GenePage from '../pages/GenePage'

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/gene/:geneQuery" element={<GenePage />} />
        <Route path="*" element={<Navigate replace to="/" />} />
      </Routes>
    </AppShell>
  )
}
